from google.cloud.billing import budgets_v1

def sample_update_budget():
    if False:
        print('Hello World!')
    client = budgets_v1.BudgetServiceClient()
    request = budgets_v1.UpdateBudgetRequest()
    response = client.update_budget(request=request)
    print(response)