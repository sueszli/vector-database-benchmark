from google.cloud.billing import budgets_v1beta1

def sample_update_budget():
    if False:
        while True:
            i = 10
    client = budgets_v1beta1.BudgetServiceClient()
    request = budgets_v1beta1.UpdateBudgetRequest()
    response = client.update_budget(request=request)
    print(response)