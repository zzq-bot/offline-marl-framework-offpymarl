import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.embed import polynomial_embed, binary_embed

class MTMADDPGCritic(nn.Module):
    def __init__(self, task2input_shape_info, 
                 task2decomposer, task2n_agents,
                 surrogate_decomposer, args):
        super(MTMADDPGCritic, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
                                       task2input_shape_info}
        
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents

        self.args = args

        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim

        self._build_attention(surrogate_decomposer)
        self._build_q()

    def forward(self, states, joint_actions, individual_inputs, task, actor_update=False):
        attn_features = self.get_attn_feature(states, joint_actions, individual_inputs, task, actor_update)
        q = self.fc(attn_features)
        return q
    
    def get_attn_feature(self, states, joint_actions, individual_inputs, task, actor_update=False):
        task_n_agents = self.task2n_agents[task]
        bs, T = states.shape[:2]
        # states: [bs, T, state_dim]
        # joint_actions = [bs, T, n_agents, n_agents*n_actions] if actor_update else [bs, T, n_agents*n_actions]
        # individual_inputs = [bs, T, n_agents, input_dim]
        # return shape (bs, T, n_agents, x)
        if actor_update:
            states = states.unsqueeze(2).repeat(1, 1, task_n_agents, 1)
            assert len(joint_actions.shape) == 4
            state_entity_embed, state_ally_embed = [], []
            for i in range(task_n_agents):
                state_entity_embed_i, state_ally_embed_i = self.get_state_plus_action_attn_feature(states[:, :, i], joint_actions[:, :, i], task, mean_ally_embed=True)
                state_entity_embed.append(state_entity_embed_i.unsqueeze(2))
                state_ally_embed.append(state_ally_embed_i.unsqueeze(2))
            state_entity_embed = th.cat(state_entity_embed, dim=2)
            state_ally_embed = th.cat(state_ally_embed, dim=2)
        else:    
            state_entity_embed, state_ally_embed = self.get_state_plus_action_attn_feature(states, joint_actions, task)
            # (bs, T, self.entity_embed_dim), (bs, T, n_agents, self.entity_embed_dim)
            state_entity_embed = state_entity_embed.unsqueeze(2).repeat(1, 1, task_n_agents, 1)
        
        state_plus_action_attn_feature = th.cat([state_entity_embed, state_ally_embed], dim=-1)
        individual_attn_feature = self.get_individual_attn_feature(individual_inputs, task)
        individual_attn_feature = individual_attn_feature.reshape(bs, T, task_n_agents, -1)

        
        return th.cat([state_plus_action_attn_feature, individual_attn_feature], dim=-1)

    def get_state_plus_action_attn_feature(self, states, joint_actions, task, mean_ally_embed=False):
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        task_n_enemies = task_decomposer.n_enemies
        task_n_actions = task_decomposer.n_actions
        task_n_entities = task_n_agents + task_n_enemies

        bs, T, _ = states.shape
        assert len(joint_actions.shape) == 3

        # decompose states
        ally_states, enemy_states, last_action_states, _ = task_decomposer.decompose_state(states)
        ally_states = th.stack(ally_states, dim=2)  # [bs, seq_len, n_agents, state_nf_al]
        enemy_states = th.stack(enemy_states, dim=2)  # [bs, seq_len, n_enemies, state_nf_en]

        # stack action information into the ally_states
        if self.state_last_action:
            last_action_states = th.stack(last_action_states, dim=2) # (bs, seq_len, n_agents, n_actions)
            _, _, last_compact_action_states = task_decomposer.decompose_action_info(last_action_states)
            ally_states = th.cat([ally_states, last_compact_action_states], dim=-1)
        
        joint_actions = joint_actions.view(bs, T, task_n_agents, task_n_actions)
        _, _, compact_action_states = task_decomposer.decompose_action_info(joint_actions) # (bs, seq_len, n_agents, no_attack_actions+1)
        ally_states = th.cat([ally_states, compact_action_states], dim=-1)
        
        # get k, q, v
        state_ally_embed = self.state_ally_encoder(ally_states) # (bs, seq_len, n_agents, entity_embed_dim)
        state_enemy_embed = self.state_enemy_encoder(enemy_states) # (bs, seq_len, n_enemies, entity_embed_dim)

        enetity_embed = th.cat([state_ally_embed, state_enemy_embed], dim=2)  # [bs, seq_len, n_entity, entity_embed_dim]

        # do attention
        proj_query = self.state_query(enetity_embed).reshape(bs * T, task_n_entities, self.attn_embed_dim)
        proj_key = self.state_key(enetity_embed).reshape(bs * T, task_n_entities, self.attn_embed_dim)
        proj_value = enetity_embed.reshape(bs * T, task_n_entities, self.entity_embed_dim)

        state_entity_embed = self._attention(proj_query, proj_key, proj_value, self.attn_embed_dim)
        state_entity_embed = state_entity_embed.mean(dim=1).reshape(bs, T, self.entity_embed_dim)
        if mean_ally_embed:
            state_ally_embed = state_ally_embed.mean(dim=2) # (bs, seq_len, entity_embed_dim)
        return state_entity_embed, state_ally_embed

    def get_individual_attn_feature(self, individual_inputs, task):
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        obs_dim = task_decomposer.obs_dim
        last_action_shape = self.task2last_action_shape[task]
        bs, T, _, _ = individual_inputs.shape
        individual_inputs = individual_inputs.reshape(bs*T*task_n_agents, -1)
       
        

        ## Build inputs/feats
        if self.args.critic_last_action:
            obs_inputs, last_action_inputs, agent_id_inputs = individual_inputs[:, :obs_dim], \
                individual_inputs[:, obs_dim:obs_dim+last_action_shape], individual_inputs[:, obs_dim+last_action_shape:]
        else:
            obs_inputs, agent_id_inputs = individual_inputs[:, :obs_dim], individual_inputs[:, obs_dim:]

        own_obs, obs_enemy_feats, obs_ally_feats = task_decomposer.decompose_obs(obs_inputs)    
        
        
        agent_id_inputs = [th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs*T, 1).to(own_obs.device)
       
        if self.args.critic_last_action:
            _, attack_action_info, compact_action_states = task_decomposer.decompose_action_info(last_action_inputs)
            # incorporate agent_id embed and compact_action_states
            own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

            attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1) #(n_enemies, bs*n_agents, 1)
            obs_enemy_feats = th.cat([th.stack(obs_enemy_feats, dim=0), attack_action_info], dim=-1).transpose(0, 1) 
        else:
            own_obs = th.cat([own_obs, agent_id_inputs], dim=-1)
            obs_enemy_feats = th.stack(obs_enemy_feats, dim=1) # (bs*T*n_agents, n_enemies, enemy_nf)
        
        obs_ally_feats = th.stack(obs_ally_feats, dim=1)

        obs_own_embed = self.obs_own_value(own_obs)
        # (bs * T * task_n_agents or bs * T, self.entity_embed_dim)

        obs_query = self.obs_query(own_obs).unsqueeze(1) # (bs*T(*task_n_agents), 1, attn_embed_dim)
        obs_ally_keys = self.obs_ally_key(obs_ally_feats)
        obs_enemy_keys = self.obs_enemy_key(obs_enemy_feats)
        obs_ally_values = self.obs_ally_value(obs_ally_feats)
        obs_enemy_values = self.obs_enemy_value(obs_enemy_feats)

        obs_ally_embed = self._attention(obs_query, obs_ally_keys, obs_ally_values, self.attn_embed_dim).squeeze(1)    
        obs_enemy_embed = self._attention(obs_query, obs_enemy_keys, obs_enemy_values, self.attn_embed_dim).squeeze(1)

        individual_attn_feature = th.cat([obs_own_embed, obs_ally_embed, obs_enemy_embed], dim=-1)
        return individual_attn_feature # (bs*T(*n_agents) attn_embed_dim)
    
    def _attention(self, q, k, v, attn_dim):
        """
            q: [bs, 1 or n_entity, attn_dim]
            k: [bs,n_entity, attn_dim]
            v: [bs, n_entity, value_dim]
        """
        energy = th.bmm(q, k.transpose(1, 2))/(attn_dim ** (1 / 2)) # (bs, 1 or n_entity, n_entity)
        score = F.softmax(energy, dim=-1)
        out = th.bmm(score, v)  # (bs, 1 or n_entity, value_dim) -> (bs, value_dim)
        return out
    
    def _build_attention(self, surrogate_decomposer):
        ####### Build State  + Joint Action Attention #######
        # Actually, State and Joint Action info can be reduced to (bs, T, n_agents, x) in critic update
        self.state_nf_al, self.state_nf_en, _ = \
            surrogate_decomposer.aligned_state_nf_al, surrogate_decomposer.aligned_state_nf_en, surrogate_decomposer.timestep_number_state_dim
        self.state_last_action, self.state_timestep_number = surrogate_decomposer.state_last_action, surrogate_decomposer.state_timestep_number
        assert not self.state_timestep_number, print("Not support timestep number state info yet")
        # get action dimension information
        self.n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
        # define state information processor
        if self.state_last_action:
            self.state_nf_al += self.n_actions_no_attack + 1
        
        self.state_nf_al += self.n_actions_no_attack + 1
        self.state_ally_encoder = nn.Linear(self.state_nf_al, self.entity_embed_dim)
        self.state_enemy_encoder = nn.Linear(self.state_nf_en, self.entity_embed_dim)
        self.state_query = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        self.state_key = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)

        # use State Info, we get global embedding, shape = (bs * seq_len, n_entities, entity_embed_dim)
        # mean-pooling -> (bs*seq_len, entity_embed_dim)
        # unsqueeze(1).repeat(1, n_agents, 1)

        ######## Build Obs Attention ########
        ###### Obs + Last_Action + Id #######
        obs_own_dim = surrogate_decomposer.aligned_own_obs_dim
        self.obs_en_dim, self.obs_al_dim = surrogate_decomposer.aligned_obs_nf_en, surrogate_decomposer.aligned_obs_nf_al 
        if self.args.critic_last_action:          
            self.wrapped_obs_own_dim = obs_own_dim + self.args.id_length + self.n_actions_no_attack + 1
            self.obs_en_dim += 1
        else:
            self.wrapped_obs_own_dim = obs_own_dim + self.args.id_length
        ## enemy_obs ought to add attack_action_infos
        

        self.obs_query = nn.Linear(self.wrapped_obs_own_dim, self.attn_embed_dim)
        self.obs_ally_key = nn.Linear(self.obs_al_dim, self.attn_embed_dim)
        self.obs_ally_value = nn.Linear(self.obs_al_dim, self.entity_embed_dim)
        self.obs_enemy_key = nn.Linear(self.obs_en_dim, self.attn_embed_dim)
        self.obs_enemy_value = nn.Linear(self.obs_en_dim, self.entity_embed_dim)
        self.obs_own_value = nn.Linear(self.wrapped_obs_own_dim, self.entity_embed_dim)
        # use Obs+last_action+Id info ->  (bs*seq_len*n_agents, entity_embed_dim*3)

    def _build_q(self):
        self.fc = nn.Sequential(
            nn.Linear(self.entity_embed_dim * 5, self.args.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, 1)
        )

        