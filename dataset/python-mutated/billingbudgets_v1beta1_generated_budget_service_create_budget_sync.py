from google.cloud.billing import budgets_v1beta1

def sample_create_budget():
    if False:
        while True:
            i = 10
    client = budgets_v1beta1.BudgetServiceClient()
    request = budgets_v1beta1.CreateBudgetRequest(parent='parent_value')
    response = client.create_budget(request=request)
    print(response)