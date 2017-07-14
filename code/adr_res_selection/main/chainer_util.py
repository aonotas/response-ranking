
def process_one(sample, xp):
    spk_agents = sample.spk_agents
    n_agent = sample.n_agents_in_lctx
    n_context = len(spk_agents)
    binned_n_agents = sample.binned_n_agents_in_ctx
    # agents_id
    agents_id = xp.full((n_agent, n_context), -1, dtype=xp.int32)
    for i, agent_idx in enumerate(spk_agents):
        agents_id[agent_idx, i] = i

    # convert to numpy
    context = [xp.array(c, dtype=xp.int32) for c in sample.context]
    response = [xp.array(r, dtype=xp.int32) for r in sample.response]

    context_length = xp.array([len(c) for c in context], dtype=xp.int32)
    response_length = xp.array([len(r) for r in response], dtype=xp.int32)

    # flatten
    context = xp.concatenate(context, axis=0)
    response = xp.concatenate(response, axis=0)

    y_adr = sample.true_adr
    y_res = sample.true_res

    item = [context, context_length, response, response_length,
            agents_id, n_agent, binned_n_agents, y_adr, y_res]
    return item


def pre_process(samples, xp):
    contexts = []
    responses = []
    contexts_length = []
    responses_length = []
    agents_ids = []
    n_agents = []
    binned_n_agents = []
    y_adr = []
    y_res = []
    for sample in samples:
        item = process_one(sample, xp)
        [_context, _context_length, _response, _response_length,
            _agents_id, _n_agent, _binned_n_agents, _y_adr, _y_res] = item

        contexts.append(_context)
        contexts_length.append(_context_length)
        responses.append(_response)
        responses_length.append(_response_length)
        agents_ids.append(_agents_id)
        n_agents.append(_n_agent)
        binned_n_agents.append(_binned_n_agents)
        y_adr.append(_y_adr)
        y_res.append(_y_res)

    # xp format
    # contexts = xp.array(contexts)
    # responses = xp.array(responses)
    # contexts_length = xp.array(contexts_length, dtype=xp.int32)
    responses_length = xp.array(responses_length, dtype=xp.int32)
    # agents_ids = xp.array(agents_ids)
    n_agents = xp.array(n_agents, dtype=xp.int32)
    binned_n_agents = xp.array(binned_n_agents, dtype=xp.int32)
    y_adr = xp.array(y_adr, dtype=xp.int32)
    y_res = xp.array(y_res, dtype=xp.int32)

    return contexts, contexts_length, responses, responses_length, agents_ids, n_agents, binned_n_agents, y_adr, y_res
