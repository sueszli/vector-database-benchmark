from google.cloud.billing import budgets_v1beta1

def sample_list_budgets():
    if False:
        while True:
            i = 10
    client = budgets_v1beta1.BudgetServiceClient()
    request = budgets_v1beta1.ListBudgetsRequest(parent='parent_value')
    page_result = client.list_budgets(request=request)
    for response in page_result:
        print(response)