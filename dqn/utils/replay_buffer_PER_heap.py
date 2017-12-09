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

        self.h2a      = -np.ones((size, 2))
        self.a2h      = -np.ones(size, int)

        self.arange = np.arange(size - 1)
        self.config = config

        P = 1.0 / (np.array(range(size - 1))+1)
        P = P ** config.alpha
        P = P/ (sum(P))

        self.values = P
        self.alpha = config.alpha

        w = (size*P)**(-config.beta)
        self.weights = w
        self.beta = config.beta

        #self.max_p = 1

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes, weights):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        #print idxes
        #print [self._encode_observation(idx + 1)[None] for idx in idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        weight_mask = weights

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, weight_mask


    def sample(self, batch_size):
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
        idexs: np.array
        """
        assert self.can_sample(batch_size)
        N = self.num_in_buffer - 1

        #print 'harvard'

        #print self.p

        #P = self.p ** self.config.alpha
        #P = P/sum(P)

        #P = self.values[rank_order]
        #P = P ** self.config.alpha
        #P = P/sum(P)
        #print P

        idxes = -np.ones(batch_size, int)
        i = 0
        while True:
            tmp_idxes = np.random.choice(self.arange, batch_size, p = self.values)
            tmp_idxes = tmp_idxes[tmp_idxes < N]
            for tmp_ind in tmp_idxes:
                if self.h2a[tmp_ind, 1] != (self.next_idx - 1) % self.size and tmp_ind not in idxes:
                    idxes[i] = tmp_ind
                    i += 1
                    if i >= batch_size:
                        break
            if i >= batch_size:
                break
        

        #print P

        #w = (N*P)**(-self.config.beta)
        #weights = w[idxes]/max(w)
        #print weights
        weights = self.weights[idxes] / self.weights[N - 1]
        true_idxes = np.int32(self.h2a[idxes, 1])

        return self._encode_sample(true_idxes, weights), true_idxes

    def update_p(self, error, idxes):
        for i in range(len(idxes)):
            true_idx, err = idxes[i], error[i]
            idx = self.a2h[true_idx]
            pre_v = self.h2a[idx, 0]
            self.h2a[idx, 0] = err
            if err > pre_v:
                self.bubble_up(idx)
            else:
                self.sink_down(idx)
        #self.max_p = max([self.max_p, max(error)])

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
        if idx >= self.size:
            idx = 0
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
            #self.p = np.empty([self.size],                     dtype=np.float32)

        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        if self.next_idx % self.config.anneal_freq == 0:
            self.alpha -= (self.config.alpha - self.config.alpha_end) / (self.config.nsteps_train / self.config.anneal_freq)
            self.alpha = min(1.0, max(0.0, self.alpha))
            P = 1.0 / (np.array(range(self.size - 1))+1)
            P = P ** self.alpha
            P = P/ (sum(P))
            self.values = P

            self.beta += (self.config.beta_end - self.config.beta) / (self.config.nsteps_train / self.config.anneal_freq)
            self.beta = max(0.0, min(1.0, self.beta))
            w = (self.size*self.values)**(-self.beta)
            self.weights = w

        return ret

    def store_effect(self, idx, action, reward, done):
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
        if self.a2h[idx] == -1:
            max_v = 1000. if idx == 0 else self.h2a[0, 0]
            self.a2h[idx] = idx
        else:
            max_v = self.h2a[0, 0]
        self.h2a[self.a2h[idx]] = [max_v, idx]
        self.bubble_up(self.a2h[idx])

        if self.next_idx % self.config.sort_freq == 0:
            idxes = np.argsort(-self.h2a[:, 0])
            self.h2a = self.h2a[idxes]
            self.a2h[np.int32(self.h2a[:, 1])] = np.arange(self.size)

        #print idx
        #print self.p
        #self.p[idx] = max(self.p)
        #print self.p[idx]
        #self.p[idx] = 1

    def bubble_up(self, idx):
        while True:
            if idx <= 0:
                break
            par_idx = (idx - 1) / 2
            tv, ti = self.h2a[idx]
            tpv, tpi = self.h2a[par_idx]
            ti, tpi = int(ti), int(tpi)
            if tv > tpv:
                self.a2h[ti] = par_idx
                self.a2h[tpi] = idx
                self.h2a[idx] = [tpv, tpi]
                self.h2a[par_idx] = [tv, ti]
                idx = par_idx
            else:
                break

    def sink_down(self, idx):
        while True:
            chi_idx1, chi1 = idx * 2 + 1, True
            chi_idx2, chi2 = idx * 2 + 2, True
            if chi_idx1 >= self.size or self.h2a[chi_idx1, 0] == -1:
                tcv1, tci1, chi1 = -1, -1, False
            else:
                tcv1, tci1 = self.h2a[chi_idx1]
            if chi_idx2 >= self.size or self.h2a[chi_idx2, 0] == -1:
                tcv2, tci2, chi2 = -1, -1, False
            else:
                tcv2, tci2 = self.h2a[chi_idx2]
            
            if not chi1 and not chi2:
                break
            chi_idx = chi_idx1 if tcv1 > tcv2 else chi_idx2

            tv, ti = self.h2a[idx]
            tcv, tci = self.h2a[chi_idx]
            ti, tci = int(ti), int(tci)
            if tv < tcv:
                self.a2h[ti] = chi_idx
                self.a2h[tci] = idx
                self.h2a[idx] = [tcv, tci]
                self.h2a[chi_idx] = [tv, ti]
                idx = chi_idx
            else:
                break