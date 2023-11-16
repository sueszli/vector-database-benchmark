from metaflow_test import MetaflowTest, ExpectationFailed, steps

class LineageTest(MetaflowTest):
    PRIORITY = 1

    @steps(0, ['start'])
    def step_start(self):
        if False:
            print('Hello World!')
        self.lineage = (self._current_step,)

    @steps(1, ['join'])
    def step_join(self):
        if False:
            for i in range(10):
                print('nop')
        self.lineage = (tuple(sorted({x.lineage for x in inputs})), self._current_step)

    @steps(2, ['all'])
    def step_all(self):
        if False:
            i = 10
            return i + 15
        self.lineage += (self._current_step,)

    def check_results(self, flow, checker):
        if False:
            print('Hello World!')
        from collections import defaultdict
        join_sets = defaultdict(set)
        lineages = {}
        graph = flow._graph

        def traverse(step, lineage):
            if False:
                print('Hello World!')
            if graph[step].type == 'join':
                join_sets[step].add(tuple(lineage))
                if len(join_sets[step]) < len(graph[step].in_funcs):
                    return
                else:
                    lineage = (tuple(sorted(join_sets[step])),)
            lineages[step] = lineage + (step,)
            for n in graph[step].out_funcs:
                traverse(n, lineage + (step,))
        traverse('start', ())
        for step in flow:
            checker.assert_artifact(step.name, 'lineage', lineages[step.name])