from google.cloud.billing import budgets_v1

def sample_delete_budget():
    if False:
        print('Hello World!')
    client = budgets_v1.BudgetServiceClient()
    request = budgets_v1.DeleteBudgetRequest(name='name_value')
    client.delete_budget(request=request)