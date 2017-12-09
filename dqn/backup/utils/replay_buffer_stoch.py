import numpy as np
import random

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    """
    Taken from Berkeley's Assignment
    """
    def __init__(self, size, frame_history_len, config):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None
        self.explore  = None
        self.sample_p = None 
        self.config   = config

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes, epsilon):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        rew_list, next_obs_list, done_list, w_list, w_e_list = [], [], [], [], []
        for num_stoch in range(self.config.num_stoch_next_obs):
            rew_array, next_obs_array, done_array, w_array, w_e_array = [], [], [], [], []
            for idx in idxes:
                #d = random.randint(self.config.stoch_window[num_stoch] + 1, self.config.stoch_window[num_stoch + 1])
                d = self.config.stoch_window[num_stoch + 1]
                '''
                if (idx + d + 1 >= self.num_in_buffer and self.num_in_buffer != self.size) or self.mark[(idx + d + 1) % self.size] != self.mark[idx]:
                    d = 0
                '''
                acc_reward, w, w_e = 0., 1., 1. / self.config.lda
                for i in range(d + 1):
                    temp_r = self.reward[(idx + i) % self.size]
                    acc_reward += temp_r * w
                    w *= self.config.gamma
                    w_e *= self.config.lda
                    if temp_r != 0.:
                        break
                    if self.done[(idx + i) % self.size]:
                        break
                    if self.explore[(idx + i + 1) % self.size]:
                        break
                    if idx + i + 2 >= self.num_in_buffer and self.num_in_buffer != self.size:
                        break
                if i <= self.config.stoch_window[num_stoch] or temp_r == 0.:
                    w_e = 0.
                rew_array.append(acc_reward)
                w_array.append(w)
                w_e_array.append(w_e)
                next_obs = self._encode_observation((idx + i + 1) % self.size)[None]
                next_obs_array.append(next_obs)
                done_array.append(1.0 if self.done[(idx + i) % self.size] else 0.0)
            next_obs_array_np = np.concatenate(next_obs_array, 0)[None]

            rew_list.append(rew_array)
            next_obs_list.append(next_obs_array_np)
            done_list.append(done_array)
            w_list.append(w_array)
            w_e_list.append(w_e_array)

        next_obs_stoch_batch = np.concatenate(next_obs_list, 0)
        done_stoch_mask      = np.array(done_list, dtype=np.float32)
        rew_stoch_batch      = np.array(rew_list, dtype=np.float32)
        w_stoch_batch        = np.array(w_list, dtype=np.float32)
        w_e_stoch_batch      = np.array(w_e_list, dtype=np.float32)
        
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, rew_stoch_batch, next_obs_stoch_batch, done_stoch_mask, w_stoch_batch, w_e_stoch_batch


    def sample(self, batch_size, epsilon):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        #idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        p = self.sample_p[:self.num_in_buffer - 1]
        p_normalized = p / np.sum(p)
        idxes = np.random.choice(self.num_in_buffer - 1, batch_size, replace=False, p=p_normalized)
        for ind in idxes:
            prev_ind = (ind - 1) % self.num_in_buffer
            if (not self.done[prev_ind]) and (self.reward[prev_ind] == 0.):
                self.sample_p[prev_ind] = self.sample_p[ind]
        self.sample_p[idxes] = self.config.epsilon_p
        idxes = list(idxes)
        return self._encode_sample(idxes, epsilon)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        # if len(self.obs.shape) <= 2:
        #     return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
            self.explore  = np.empty([self.size],                     dtype=np.float32)
            self.sample_p = np.empty([self.size],                     dtype=np.float32)
            self.mark     = np.ones([self.size],                      dtype=np.float32)
            self.cur_mark = np.random.rand()
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done, explore):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done
        self.explore[idx] = explore
        if reward == 0.:
            self.sample_p[idx] = self.config.epsilon_p
        elif reward < 0.:
            self.sample_p[idx] = self.config.alpha_p
        else:
            self.sample_p[idx] = self.config.beta_p
        self.mark[idx] = self.cur_mark
        if done:
            self.cur_mark = np.random.rand()
            self.sample_p[idx] = max(self.sample_p[idx], self.config.epsilon_p)
