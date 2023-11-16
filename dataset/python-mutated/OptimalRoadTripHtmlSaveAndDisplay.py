"""
Randy Olson's Shortest Route Program modified By Andrew Liesinger to:
    1: Detect waypoints file at runtime - if found use it, otherwise look up distances via google calls (and then save to waypoint file)
    2: Dynamically create and open an HTML file showing the route when a shorter route is found
    3: Make it easier to tinker with the Generation / Population parameters
"""
from __future__ import print_function
from itertools import combinations
import googlemaps
import pandas as pd
import numpy as np
import os.path
import random
import webbrowser
GOOGLE_MAPS_API_KEY = 'Your Key Here'
waypoints_file = 'my-waypoints-dist-dur.tsv'
output_file = 'Output.html'
thisRunGenerations = 5000
thisRunPopulation_size = 100
all_waypoints = ['USS Alabama, Battleship Parkway, Mobile, AL', 'Grand Canyon National Park, Arizona', 'Toltec Mounds, Scott, AR', 'San Andreas Fault, San Benito County, CA', 'Cable Car Museum, 94108, 1201 Mason St, San Francisco, CA 94108', 'Pikes Peak, Colorado', 'The Mark Twain House & Museum, Farmington Avenue, Hartford, CT', 'New Castle Historic District, Delaware', 'White House, Pennsylvania Avenue Northwest, Washington, DC', 'Cape Canaveral, FL', 'Okefenokee Swamp Park, Okefenokee Swamp Park Road, Waycross, GA', 'Craters of the Moon National Monument & Preserve, Arco, ID', 'Lincoln Home National Historic Site Visitor Center, 426 South 7th Street, Springfield, IL', 'West Baden Springs Hotel, West Baden Avenue, West Baden Springs, IN', 'Terrace Hill, Grand Avenue, Des Moines, IA', 'C. W. Parker Carousel Museum, South Esplanade Street, Leavenworth, KS', 'Mammoth Cave National Park, Mammoth Cave Pkwy, Mammoth Cave, KY', 'French Quarter, New Orleans, LA', 'Acadia National Park, Maine', 'Maryland State House, 100 State Cir, Annapolis, MD 21401', 'USS Constitution, Boston, MA', 'Olympia Entertainment, Woodward Avenue, Detroit, MI', 'Fort Snelling, Tower Avenue, Saint Paul, MN', 'Vicksburg National Military Park, Clay Street, Vicksburg, MS', 'Gateway Arch, Washington Avenue, St Louis, MO', 'Glacier National Park, West Glacier, MT', 'Ashfall Fossil Bed, Royal, NE', 'Hoover Dam, NV', 'Omni Mount Washington Resort, Mount Washington Hotel Road, Bretton Woods, NH', 'Congress Hall, Congress Place, Cape May, NJ 08204', 'Carlsbad Caverns National Park, Carlsbad, NM', 'Statue of Liberty, Liberty Island, NYC, NY', 'Wright Brothers National Memorial Visitor Center, Manteo, NC', 'Fort Union Trading Post National Historic Site, Williston, North Dakota 1804, ND', 'Spring Grove Cemetery, Spring Grove Avenue, Cincinnati, OH', 'Chickasaw National Recreation Area, 1008 W 2nd St, Sulphur, OK 73086', 'Columbia River Gorge National Scenic Area, Oregon', 'Liberty Bell, 6th Street, Philadelphia, PA', 'The Breakers, Ochre Point Avenue, Newport, RI', "Fort Sumter National Monument, Sullivan's Island, SC", 'Mount Rushmore National Memorial, South Dakota 244, Keystone, SD', 'Graceland, Elvis Presley Boulevard, Memphis, TN', 'The Alamo, Alamo Plaza, San Antonio, TX', 'Bryce Canyon National Park, Hwy 63, Bryce, UT', 'Shelburne Farms, Harbor Road, Shelburne, VT', 'Mount Vernon, Fairfax County, Virginia', 'Hanford Site, Benton County, WA', 'Lost World Caverns, Lewisburg, WV', 'Taliesin, County Road C, Spring Green, Wisconsin', 'Yellowstone National Park, WY 82190']

def CreateOptimalRouteHtmlFile(optimal_route, distance, display=True):
    if False:
        for i in range(10):
            print('nop')
    optimal_route = list(optimal_route)
    optimal_route += [optimal_route[0]]
    Page_1 = '\n    <!DOCTYPE html>\n    <html lang="en">\n      <head>\n        <meta charset="utf-8">\n        <meta name="viewport" content="initial-scale=1.0, user-scalable=no">\n        <meta name="description" content="Randy Olson uses machine learning to find the optimal road trip across the U.S.">\n        <meta name="author" content="Randal S. Olson">\n        \n        <title>The optimal road trip across the U.S. according to machine learning</title>\n        <style>\n          html, body, #map-canvas {\n            height: 100%;\n            margin: 0px;\n            padding: 0px\n          }\n          #panel {\n            position: absolute;\n            top: 5px;\n            left: 50%;\n            margin-left: -180px;\n            z-index: 5;\n            background-color: #fff;\n            padding: 10px;\n            border: 1px solid #999;\n          }\n        </style>\n        <script src="https://maps.googleapis.com/maps/api/js?v=3.exp&signed_in=true"></script>\n        <script>\n            var routes_list = []\n            var markerOptions = {icon: "http://maps.gstatic.com/mapfiles/markers2/marker.png"};\n            var directionsDisplayOptions = {preserveViewport: true,\n                                            markerOptions: markerOptions};\n            var directionsService = new google.maps.DirectionsService();\n            var map;\n\n            function initialize() {\n              var center = new google.maps.LatLng(39, -96);\n              var mapOptions = {\n                zoom: 5,\n                center: center\n              };\n              map = new google.maps.Map(document.getElementById(\'map-canvas\'), mapOptions);\n              for (i=0; i<routes_list.length; i++) {\n                routes_list[i].setMap(map); \n              }\n            }\n\n            function calcRoute(start, end, routes) {\n              \n              var directionsDisplay = new google.maps.DirectionsRenderer(directionsDisplayOptions);\n\n              var waypts = [];\n              for (var i = 0; i < routes.length; i++) {\n                waypts.push({\n                  location:routes[i],\n                  stopover:true});\n                }\n              \n              var request = {\n                  origin: start,\n                  destination: end,\n                  waypoints: waypts,\n                  optimizeWaypoints: false,\n                  travelMode: google.maps.TravelMode.DRIVING\n              };\n\n              directionsService.route(request, function(response, status) {\n                if (status == google.maps.DirectionsStatus.OK) {\n                    directionsDisplay.setDirections(response);      \n                }\n              });\n\n              routes_list.push(directionsDisplay);\n            }\n\n            function createRoutes(route) {\n                // Google\'s free map API is limited to 10 waypoints so need to break into batches\n                route.push(route[0]);\n                var subset = 0;\n                while (subset < route.length) {\n                    var waypointSubset = route.slice(subset, subset + 10);\n\n                    var startPoint = waypointSubset[0];\n                    var midPoints = waypointSubset.slice(1, waypointSubset.length - 1);\n                    var endPoint = waypointSubset[waypointSubset.length - 1];\n\n                    calcRoute(startPoint, endPoint, midPoints);\n\n                    subset += 9;\n                }\n            }\n    '
    Page_2 = '\n            \n            createRoutes(optimal_route);\n\n            google.maps.event.addDomListener(window, \'load\', initialize);\n\n        </script>\n      </head>\n      <body>\n        <div id="map-canvas"></div>\n      </body>\n    </html>\n    '
    localoutput_file = output_file.replace('.html', '_' + str(distance) + '.html')
    with open(localoutput_file, 'w') as fs:
        fs.write(Page_1)
        fs.write('\t\t\toptimal_route = {0}'.format(str(optimal_route)))
        fs.write(Page_2)
    if display:
        webbrowser.open_new_tab(localoutput_file)

def compute_fitness(solution):
    if False:
        print('Hello World!')
    '\n        This function returns the total distance traveled on the current road trip.\n        \n        The genetic algorithm will favor road trips that have shorter\n        total distances traveled.\n    '
    solution_fitness = 0.0
    for index in range(len(solution)):
        waypoint1 = solution[index - 1]
        waypoint2 = solution[index]
        solution_fitness += waypoint_distances[frozenset([waypoint1, waypoint2])]
    return solution_fitness

def generate_random_agent():
    if False:
        for i in range(10):
            print('nop')
    '\n        Creates a random road trip from the waypoints.\n    '
    new_random_agent = list(all_waypoints)
    random.shuffle(new_random_agent)
    return tuple(new_random_agent)

def mutate_agent(agent_genome, max_mutations=3):
    if False:
        return 10
    '\n        Applies 1 - `max_mutations` point mutations to the given road trip.\n        \n        A point mutation swaps the order of two waypoints in the road trip.\n    '
    agent_genome = list(agent_genome)
    num_mutations = random.randint(1, max_mutations)
    for mutation in range(num_mutations):
        swap_index1 = random.randint(0, len(agent_genome) - 1)
        swap_index2 = swap_index1
        while swap_index1 == swap_index2:
            swap_index2 = random.randint(0, len(agent_genome) - 1)
        (agent_genome[swap_index1], agent_genome[swap_index2]) = (agent_genome[swap_index2], agent_genome[swap_index1])
    return tuple(agent_genome)

def shuffle_mutation(agent_genome):
    if False:
        while True:
            i = 10
    '\n        Applies a single shuffle mutation to the given road trip.\n        \n        A shuffle mutation takes a random sub-section of the road trip\n        and moves it to another location in the road trip.\n    '
    agent_genome = list(agent_genome)
    start_index = random.randint(0, len(agent_genome) - 1)
    length = random.randint(2, 20)
    genome_subset = agent_genome[start_index:start_index + length]
    agent_genome = agent_genome[:start_index] + agent_genome[start_index + length:]
    insert_index = random.randint(0, len(agent_genome) + len(genome_subset) - 1)
    agent_genome = agent_genome[:insert_index] + genome_subset + agent_genome[insert_index:]
    return tuple(agent_genome)

def generate_random_population(pop_size):
    if False:
        return 10
    '\n        Generates a list with `pop_size` number of random road trips.\n    '
    random_population = []
    for agent in range(pop_size):
        random_population.append(generate_random_agent())
    return random_population

def run_genetic_algorithm(generations=5000, population_size=100):
    if False:
        for i in range(10):
            print('nop')
    '\n        The core of the Genetic Algorithm.\n        \n        `generations` and `population_size` must be a multiple of 10.\n    '
    current_best_distance = -1
    population_subset_size = int(population_size / 10.0)
    generations_10pct = int(generations / 10.0)
    population = generate_random_population(population_size)
    for generation in range(generations):
        population_fitness = {}
        for agent_genome in population:
            if agent_genome in population_fitness:
                continue
            population_fitness[agent_genome] = compute_fitness(agent_genome)
        new_population = []
        for (rank, agent_genome) in enumerate(sorted(population_fitness, key=population_fitness.get)[:population_subset_size]):
            if (generation % generations_10pct == 0 or generation == generations - 1) and rank == 0:
                current_best_genome = agent_genome
                print('Generation %d best: %d | Unique genomes: %d' % (generation, population_fitness[agent_genome], len(population_fitness)))
                print(agent_genome)
                print('')
                if population_fitness[agent_genome] < current_best_distance or current_best_distance < 0:
                    current_best_distance = population_fitness[agent_genome]
                    CreateOptimalRouteHtmlFile(agent_genome, current_best_distance, False)
            new_population.append(agent_genome)
            for offspring in range(2):
                new_population.append(mutate_agent(agent_genome, 3))
            for offspring in range(7):
                new_population.append(shuffle_mutation(agent_genome))
        for i in range(len(population))[::-1]:
            del population[i]
        population = new_population
    return current_best_genome
if __name__ == '__main__':
    print('Begin finding shortest route')
    file_path = waypoints_file
    if os.path.exists(file_path):
        print('Waypoints exist')
        waypoint_distances = {}
        waypoint_durations = {}
        all_waypoints = set()
        waypoint_data = pd.read_csv(file_path, sep='\t')
        for (i, row) in waypoint_data.iterrows():
            waypoint_distances[frozenset([row.waypoint1, row.waypoint2])] = row.distance_m
            waypoint_durations[frozenset([row.waypoint1, row.waypoint2])] = row.duration_s
            all_waypoints.update([row.waypoint1, row.waypoint2])
    else:
        print('Collecting Waypoints')
        waypoint_distances = {}
        waypoint_durations = {}
        gmaps = googlemaps.Client(GOOGLE_MAPS_API_KEY)
        for (waypoint1, waypoint2) in combinations(all_waypoints, 2):
            try:
                route = gmaps.distance_matrix(origins=[waypoint1], destinations=[waypoint2], mode='driving', language='English', units='metric')
                distance = route['rows'][0]['elements'][0]['distance']['value']
                duration = route['rows'][0]['elements'][0]['duration']['value']
                waypoint_distances[frozenset([waypoint1, waypoint2])] = distance
                waypoint_durations[frozenset([waypoint1, waypoint2])] = duration
            except Exception as e:
                print('Error with finding the route between %s and %s.' % (waypoint1, waypoint2))
        print('Saving Waypoints')
        with open(waypoints_file, 'w') as out_file:
            out_file.write('\t'.join(['waypoint1', 'waypoint2', 'distance_m', 'duration_s']))
            for (waypoint1, waypoint2) in waypoint_distances.keys():
                out_file.write('\n' + '\t'.join([waypoint1, waypoint2, str(waypoint_distances[frozenset([waypoint1, waypoint2])]), str(waypoint_durations[frozenset([waypoint1, waypoint2])])]))
    print('Search for optimal route')
    optimal_route = run_genetic_algorithm(generations=thisRunGenerations, population_size=thisRunPopulation_size)
    CreateOptimalRouteHtmlFile(optimal_route, 1, True)