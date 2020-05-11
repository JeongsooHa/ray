import logging
import collections
import numpy as np

import ray
from ray.rllib.optimizers.replay_buffer import ReplayBuffer, \
    PrioritizedReplayBuffer
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.compression import pack_if_needed
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.utils.memory import ray_get_and_free

logger = logging.getLogger(__name__)


class SyncReplayOptimizer(PolicyOptimizer):
    """Variant of the local sync optimizer that supports replay (for DQN).

    This optimizer requires that rollout workers return an additional
    "td_error" array in the info return of compute_gradients(). This error
    term will be used for sample prioritization."""

    def __init__(
        self,
        workers,
        learning_starts=1000,
        buffer_size=10000,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta=0.4,
        prioritized_replay_eps=1e-6,
        final_prioritized_replay_beta=0.4,
        train_batch_size=32,
        before_learn_on_batch=None,
        synchronize_sampling=False,
        prioritized_replay_beta_annealing_timesteps=100000 * 0.2,
    ):
        """Initialize an sync replay optimizer.

        Args:
            workers (WorkerSet): all workers
            learning_starts (int): wait until this many steps have been sampled
                before starting optimization.
            buffer_size (int): max size of the replay buffer
            prioritized_replay (bool): whether to enable prioritized replay
            prioritized_replay_alpha (float): replay alpha hyperparameter
            prioritized_replay_beta (float): replay beta hyperparameter
            prioritized_replay_eps (float): replay eps hyperparameter
            final_prioritized_replay_beta (float): Final value of beta.
            train_batch_size (int): size of batches to learn on
            before_learn_on_batch (function): callback to run before passing
                the sampled batch to learn on
            synchronize_sampling (bool): whether to sample the experiences for
                all policies with the same indices (used in MADDPG).
            prioritized_replay_beta_annealing_timesteps (int): The timestep at
                which PR-beta annealing should end.
        """
        PolicyOptimizer.__init__(self, workers)

        self.replay_starts = learning_starts

        # Linearly annealing beta used in Rainbow paper, stopping at
        # `final_prioritized_replay_beta`.
        self.prioritized_replay_beta = PiecewiseSchedule(
            endpoints=[(0, prioritized_replay_beta),
                       (prioritized_replay_beta_annealing_timesteps,
                        final_prioritized_replay_beta)],
            outside_value=final_prioritized_replay_beta,
            framework=None)
        self.prioritized_replay_eps = prioritized_replay_eps
        self.train_batch_size = train_batch_size
        self.before_learn_on_batch = before_learn_on_batch
        self.synchronize_sampling = synchronize_sampling

        # Stats
        self.update_weights_timer = TimerStat()
        self.sample_timer = TimerStat()
        self.replay_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.learner_stats = {}

        # Set up replay buffer
        if prioritized_replay:

            def new_buffer():
                return PrioritizedReplayBuffer(
                    buffer_size, alpha=prioritized_replay_alpha)
        else:

            def new_buffer():
                return ReplayBuffer(buffer_size)

        # def new_temp_buffer():
        #     return ReplayBuffer(temp_buffer_size)

        self.replay_buffers = collections.defaultdict(new_buffer)
        # self.temp_replay_buffers = collections.defaultdict(new_temp_buffer)
        self.temp_replay_buffers = {}
        self.buffer_countor = {}
        self.init = True
        self.num_agents = 0

        if buffer_size < self.replay_starts:
            logger.warning("buffer_size={} < replay_starts={}".format(
                buffer_size, self.replay_starts))
        self.debug_print = False

    @override(PolicyOptimizer)
    def step(self):
        if self.debug_print:
            print("##### Call step function in DQN")
        with self.update_weights_timer:
            if self.workers.remote_workers():
                weights = ray.put(self.workers.local_worker().get_weights())
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)
        if self.debug_print:
            print("##### finished set_weights")
        with self.sample_timer:
            if self.workers.remote_workers():
                batch = SampleBatch.concat_samples(
                    ray_get_and_free([
                        e.sample.remote()
                        for e in self.workers.remote_workers()
                    ]))

            else:
                batch = self.workers.local_worker().sample()
            # Handle everything as if multiagent
            if isinstance(batch, SampleBatch):
                batch = MultiAgentBatch({
                    DEFAULT_POLICY_ID: batch
                }, batch.count)

            if self.init:
                for policy_id, s in batch.policy_batches.items():
                    self.num_agents += 1
                    self.temp_replay_buffers[policy_id] = []
                    self.buffer_countor[policy_id] = 0
                self.init = False

            # import ipdb; ipdb.set_trace()
            for policy_id, s in batch.policy_batches.items():
                for row in s.rows():
                    trajectory = self.input_data_and_check_packetid(policy_id, row)
                    if trajectory is not None:
                        # put data into original buffer if length of temp RB is 20
                        if self.debug_print:
                            print("Origin replay buffer  packID", trajectory["infos"]["packetid"][0], "reward",
                                  trajectory["rewards"], "delivery", trajectory["infos"]["delivery"][0])
                        self.replay_buffers[policy_id].add(
                            trajectory["obs"],
                            trajectory["actions"],
                            trajectory["rewards"],
                            trajectory["new_obs"],
                            trajectory["dones"],
                            weight=None)
                        self.buffer_countor[policy_id] += 1

            # for policy_id, s in batch.policy_batches.items():
            #     for row in s.rows():
            #         self.temp_replay_buffers[policy_id] = (
            #             pack_if_needed(row["obs"]),
            #             row["actions"],
            #             row["rewards"],
            #             pack_if_needed(row["new_obs"]),
            #             row["dones"],
            #             weight=None)
            #         check = self.temp_replay_buffers[policy_id].checkpacket(row)
            #         if check:
            #             self.temp_replay_buffers[policy_id].move_to_origin_buffer(row, self.replay_buffer)
            #
            #         self.replay_buffers[policy_id].add(
            #             pack_if_needed(row["obs"]),
            #             row["actions"],
            #             row["rewards"],
            #             pack_if_needed(row["new_obs"]),
            #             row["dones"],
            #             weight=None)

            # for policy_id, s in batch.policy_batches.items():
            #     for row in s.rows():
            #         self.replay_buffers[policy_id].add(
            #             pack_if_needed(row["obs"]),
            #             row["actions"],
            #             row["rewards"],
            #             pack_if_needed(row["new_obs"]),
            #             row["dones"],
            #             weight=None)
        # if self.num_steps_sampled >= self.replay_starts:

        # If the minimum number in the buffer is self.replay_starts or more
        if min(self.buffer_countor.values()) >= self.replay_starts:
            self._optimize()

        self.num_steps_sampled += batch.count

    def input_data_and_check_packetid(self, policy_id, row):

        # Check busy node. If the agent is busy, packetid is -1.
        if row["infos"]["packetid"][0] == -1:
            return None
        else:
            # obs = ["C", "H"]
            # infos = ["delivery", "nACKs", "packet id"]
            # Check delivery flag. If the packet is delivered, delivery is 1.
            # We have to check if that packetid is in temp_replay_buffer.
            if row["infos"]["delivery"][0] == 1:
                if self.debug_print:
                    print("##### delivery == 1 #####")
                # Find same packet id
                # Check if there is the same packet id in temp_repaly_buffer.
                for agent_id in self.temp_replay_buffers.keys():
                    for trajectory in self.temp_replay_buffers[agent_id]:
                        if (trajectory["infos"]["packetid"][0] == row["infos"]["packetid"][0]) and \
                                (trajectory["infos"]["delivery"][0] != 1) :
                            # Change rewards and delivery flag
                            trajectory["rewards"] += self.num_agents
                            trajectory["infos"]["delivery"][0] = 1
                            if self.debug_print:
                                print("##### UPDATE REWARD #####\nTEMP", agent_id, "reward ", trajectory["rewards"], " packetid ", trajectory["infos"]["packetid"][0])

            else:
                # Put data into temp_replay_buffers if delivery flag is not 1.
                self.temp_replay_buffers[policy_id].append(row)
                if self.debug_print:
                    for agent_i in self.temp_replay_buffers.keys():
                        print("TEMP", agent_i)
                        print("reward\t\t", " delivery\t", " packetid")
                        for traj in self.temp_replay_buffers[agent_i]:
                            print("   ", traj["rewards"], "\t\t   ", traj["infos"]["delivery"][0], "\t\t   ", traj["infos"]["packetid"][0])
                        else:
                            print("")
        # If length of steps is more than 20
        if self.num_steps_sampled > 1000:
            try:
                return self.temp_replay_buffers[policy_id].pop(0)
            except Exception:
                # If there is no data in temp_replay_buffer
                return None
        else:
            return None

    @override(PolicyOptimizer)
    def stats(self):
        return dict(
            PolicyOptimizer.stats(self), **{
                "sample_time_ms": round(1000 * self.sample_timer.mean, 3),
                "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
                "grad_time_ms": round(1000 * self.grad_timer.mean, 3),
                "update_time_ms": round(1000 * self.update_weights_timer.mean,
                                        3),
                "opt_peak_throughput": round(self.grad_timer.mean_throughput,
                                             3),
                "opt_samples": round(self.grad_timer.mean_units_processed, 3),
                "learner": self.learner_stats,
            })

    def _optimize(self):
        samples = self._replay()

        with self.grad_timer:
            if self.before_learn_on_batch:
                samples = self.before_learn_on_batch(
                    samples,
                    self.workers.local_worker().policy_map,
                    self.train_batch_size)
            info_dict = self.workers.local_worker().learn_on_batch(samples)
            for policy_id, info in info_dict.items():
                self.learner_stats[policy_id] = get_learner_stats(info)
                replay_buffer = self.replay_buffers[policy_id]
                if isinstance(replay_buffer, PrioritizedReplayBuffer):
                    td_error = info["td_error"]
                    new_priorities = (
                        np.abs(td_error) + self.prioritized_replay_eps)
                    replay_buffer.update_priorities(
                        samples.policy_batches[policy_id]["batch_indexes"],
                        new_priorities)
            self.grad_timer.push_units_processed(samples.count)

        self.num_steps_trained += samples.count

    def _replay(self):
        samples = {}
        idxes = None
        with self.replay_timer:
            for policy_id, replay_buffer in self.replay_buffers.items():
                if self.synchronize_sampling:
                    if idxes is None:
                        idxes = replay_buffer.sample_idxes(
                            self.train_batch_size)
                else:
                    idxes = replay_buffer.sample_idxes(self.train_batch_size)

                if isinstance(replay_buffer, PrioritizedReplayBuffer):
                    (obses_t, actions, rewards, obses_tp1, dones, weights,
                     batch_indexes) = replay_buffer.sample_with_idxes(
                        idxes,
                        beta=self.prioritized_replay_beta.value(
                            self.num_steps_trained))
                else:
                    (obses_t, actions, rewards, obses_tp1,
                     dones) = replay_buffer.sample_with_idxes(idxes)
                    weights = np.ones_like(rewards)
                    batch_indexes = -np.ones_like(rewards)
                samples[policy_id] = SampleBatch({
                    "obs": obses_t,
                    "actions": actions,
                    "rewards": rewards,
                    "new_obs": obses_tp1,
                    "dones": dones,
                    "weights": weights,
                    "batch_indexes": batch_indexes
                })
        return MultiAgentBatch(samples, self.train_batch_size)
