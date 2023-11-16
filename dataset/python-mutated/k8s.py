from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.operators.bash import BashOperator
if TYPE_CHECKING:
    from airflow.models.operator import Operator

def get_describe_pod_operator(cluster_name: str, pod_name: str) -> Operator:
    if False:
        print('Hello World!')
    "Returns an operator that'll print the output of a `k describe pod` in the airflow logs."
    return BashOperator(task_id='describe_pod', bash_command=f'\n                install_aws.sh;\n                install_kubectl.sh;\n                # configure kubectl to hit the right cluster\n                aws eks update-kubeconfig --name {cluster_name};\n                # once all this setup is done, actually describe the pod\n                echo "vvv pod description below vvv";\n                kubectl describe pod {pod_name};\n                echo "^^^ pod description above ^^^" ')