from google.cloud.billing import budgets_v1beta1

def sample_delete_budget():
    if False:
        return 10
    client = budgets_v1beta1.BudgetServiceClient()
    request = budgets_v1beta1.DeleteBudgetRequest(name='name_value')
    client.delete_budget(request=request)