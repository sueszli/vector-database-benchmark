from google.cloud import compute_v1

def sample_validate():
    if False:
        i = 10
        return i + 15
    client = compute_v1.UrlMapsClient()
    request = compute_v1.ValidateUrlMapRequest(project='project_value', url_map='url_map_value')
    response = client.validate(request=request)
    print(response)