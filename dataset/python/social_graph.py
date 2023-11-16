import networkx as nx
from django.db import connection

from api.models import Person


def build_social_graph(user):
    query = """
        with face as (
            select photo_id, person_id, name, owner_id
            from api_face join api_person on api_person.id = person_id
                          join api_photo on api_photo.image_hash = photo_id
            where person_label_is_inferred = false
                and owner_id = {}
        )
        select f1.name, f2.name
        from face f1 join face f2 using (photo_id)
        where f1.person_id != f2.person_id
        group by f1.name, f2.name
    """.replace(
        "{}", str(user.id)
    )
    G = nx.Graph()
    with connection.cursor() as cursor:
        cursor.execute(query)
        links = cursor.fetchall()
        if len(links) == 0:
            return {"nodes": [], "links": []}
        for link in links:
            G.add_edge(link[0], link[1])
    pos = nx.spring_layout(G, k=1 / 2, scale=1000, iterations=20)
    return {
        "nodes": [{"id": node, "x": pos[0], "y": pos[1]} for node, pos in pos.items()],
        "links": [{"source": pair[0], "target": pair[1]} for pair in G.edges()],
    }


def build_ego_graph(person_id):
    G = nx.Graph()
    person = Person.objects.prefetch_related("faces__photo__faces__person").filter(
        id=person_id
    )[0]
    for this_person_face in person.faces.all():
        for other_person_face in this_person_face.photo.faces.all():
            G.add_edge(person.name, other_person_face.person.name)
    nodes = [{"id": node} for node in G.nodes()]
    links = [{"source": pair[0], "target": pair[1]} for pair in G.edges()]
    res = {"nodes": nodes, "links": links}
    return res
