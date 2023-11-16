"""Examples for RecSim envs ready to be used by RLlib Algorithms.

RecSim is a configurable recommender systems simulation platform.
Source: https://github.com/google-research/recsim
"""
from recsim import choice_model
from recsim.environments import long_term_satisfaction as lts, interest_evolution as iev, interest_exploration as iex
from ray.rllib.env.wrappers.recsim import make_recsim_env
from ray.tune import register_env

def lts_user_model_creator(env_ctx):
    if False:
        return 10
    return lts.LTSUserModel(env_ctx['slate_size'], user_state_ctor=lts.LTSUserState, response_model_ctor=lts.LTSResponse)

def lts_document_sampler_creator(env_ctx):
    if False:
        print('Hello World!')
    return lts.LTSDocumentSampler()
LongTermSatisfactionRecSimEnv = make_recsim_env(recsim_user_model_creator=lts_user_model_creator, recsim_document_sampler_creator=lts_document_sampler_creator, reward_aggregator=lts.clicked_engagement_reward)

def iex_user_model_creator(env_ctx):
    if False:
        i = 10
        return i + 15
    return iex.IEUserModel(env_ctx['slate_size'], user_state_ctor=iex.IEUserState, response_model_ctor=iex.IEResponse, seed=env_ctx['seed'])

def iex_document_sampler_creator(env_ctx):
    if False:
        for i in range(10):
            print('nop')
    return iex.IETopicDocumentSampler(seed=env_ctx['seed'])
InterestExplorationRecSimEnv = make_recsim_env(recsim_user_model_creator=iex_user_model_creator, recsim_document_sampler_creator=iex_document_sampler_creator, reward_aggregator=iex.total_clicks_reward)

def iev_user_model_creator(env_ctx):
    if False:
        print('Hello World!')
    return iev.IEvUserModel(env_ctx['slate_size'], choice_model_ctor=choice_model.MultinomialProportionalChoiceModel, response_model_ctor=iev.IEvResponse, user_state_ctor=iev.IEvUserState, seed=env_ctx['seed'])

class SingleClusterIEvVideo(iev.IEvVideo):

    def __init__(self, doc_id, features, video_length=None, quality=None):
        if False:
            while True:
                i = 10
        super(SingleClusterIEvVideo, self).__init__(doc_id=doc_id, features=features, cluster_id=0, video_length=video_length, quality=quality)

def iev_document_sampler_creator(env_ctx):
    if False:
        return 10
    return iev.UtilityModelVideoSampler(doc_ctor=iev.IEvVideo, seed=env_ctx['seed'])
InterestEvolutionRecSimEnv = make_recsim_env(recsim_user_model_creator=iev_user_model_creator, recsim_document_sampler_creator=iev_document_sampler_creator, reward_aggregator=iev.clicked_watchtime_reward)
register_env(name='RecSim-v1', env_creator=lambda env_ctx: InterestEvolutionRecSimEnv(env_ctx))