"""
This class takes in a Graph object and creates interactive visualisations using Plotly's Dash.
The DashGraph class contains private functions used to generate the frontend components needed to create the UI.

Running run_server() will produce the warning "Warning: This is a development server. Do not use app.run_server
in production, use a production WSGI server like gunicorn instead.".
However, this is okay and the Dash server will run without a problem.
"""
import json
import random
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from networkx import nx

class DashGraph:
    """
    This DashGraph class creates a server for Dash cytoscape visualisations.
    """

    def __init__(self, input_graph, app_display='default'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Initialises the DashGraph object from the Graph class object.\n        Dash creates a mini Flask server to visualise the graphs.\n\n        :param app_display: (str) 'default' by default and 'jupyter notebook' for running Dash inside Jupyter Notebook.\n        :param input_graph: (Graph) Graph class from graph.py.\n        "
        pass

    def _set_cyto_graph(self):
        if False:
            return 10
        '\n        Sets the cytoscape graph elements.\n        '
        pass

    def _get_node_group(self, node_name):
        if False:
            while True:
                i = 10
        '\n        Returns the industry or sector name for a given node name.\n\n        :param node_name: (str) Name of a given node in the graph.\n        :return: (str) Name of industry that the node is in or "default" for nodes which haven\'t been assigned a group.\n        '
        pass

    def _get_node_size(self, index):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the node size for given node index if the node sizes have been set.\n\n        :param index: (int) The index of the node.\n        :return: (float) Returns size of node set, 0 if it has not been set.\n        '
        pass

    def _update_elements(self, dps=4):
        if False:
            for i in range(10):
                print('nop')
        '\n        Updates the elements needed for the Dash Cytoscape Graph object.\n\n        :param dps: (int) Decimal places to round the edge values.\n        '
        pass

    def _generate_layout(self):
        if False:
            print('Hello World!')
        '\n        Generates the layout for cytoscape.\n\n        :return: (dbc.Container) Returns Dash Bootstrap Component Container containing the layout of UI.\n        '
        pass

    def _assign_colours_to_groups(self, groups):
        if False:
            return 10
        '\n        Assigns the colours to industry or sector groups by creating a dictionary of group name to colour.\n\n        :param groups: (List) List of industry groups as strings.\n        '
        pass

    def _style_colours(self):
        if False:
            print('Hello World!')
        '\n        Appends the colour styling to stylesheet for the different groups.\n        '
        pass

    def _assign_sizes(self):
        if False:
            while True:
                i = 10
        '\n        Assigns the node sizing by appending to the stylesheet.\n        '
    pass

    def get_server(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a small Flask server.\n\n        :return: (Dash) Returns the Dash app object, which can be run using run_server.\n            Returns a Jupyter Dash object if DashGraph has been initialised for Jupyter Notebook.\n        '
        pass

    @staticmethod
    def _update_cytoscape_layout(layout):
        if False:
            return 10
        "\n        Callback function for updating the cytoscape layout.\n        The useful layouts for MST have been included as options (cola, cose-bilkent, spread).\n\n        :return: (Dict) Dictionary of the key 'name' to the desired layout (e.g. cola, spread).\n        "
        pass

    def _update_stat_json(self, stat_name):
        if False:
            i = 10
            return i + 15
        '\n        Callback function for updating the statistic shown.\n\n        :param stat_name: (str) Name of the statistic to display (e.g. graph_summary).\n        :return: (json) Json of the graph information depending on chosen statistic.\n        '
        pass

    def get_graph_summary(self):
        if False:
            while True:
                i = 10
        '\n        Returns the Graph Summary statistics.\n        The following statistics are included - the number of nodes and edges, smallest and largest edge,\n        average node connectivity, normalised tree length and the average shortest path.\n\n        :return: (Dict) Dictionary of graph summary statistics.\n        '
        pass

    def _round_decimals(self, dps):
        if False:
            print('Hello World!')
        '\n        Callback function for updating decimal places.\n        Updates the elements to modify the rounding of edge values.\n\n        :param dps: (int) Number of decimals places to round to.\n        :return: (List) Returns the list of elements used to define graph.\n        '
        pass

    def _get_default_stylesheet(self):
        if False:
            return 10
        '\n        Returns the default stylesheet for initialisation.\n\n        :return: (List) A List of definitions used for Dash styling.\n        '
        pass

    def _get_toast(self):
        if False:
            while True:
                i = 10
        '\n        Toast is the floating colour legend to display when industry groups have been added.\n        This method returns the toast component with the styled colour legend.\n\n        :return: (html.Div) Returns Div containing colour legend.\n        '
        pass

    def _get_default_controls(self):
        if False:
            print('Hello World!')
        '\n        Returns the default controls for initialisation.\n\n        :return: (dbc.Card) Dash Bootstrap Component Card which defines the side panel.\n        '
        pass

class PMFGDash(DashGraph):
    """
    PMFGDash class, a child of DashGraph, is the Dash interface class to display the PMFG.
    """

    def __init__(self, input_graph, app_display='default'):
        if False:
            return 10
        '\n        Initialise the PMFGDash class but override the layout options.\n        '
        pass

    def _update_elements(self, dps=4):
        if False:
            i = 10
            return i + 15
        "\n        Overrides the parent DashGraph class method _update_elements, to add styling for the MST edges.\n        Updates the elements needed for the Dash Cytoscape Graph object.\n\n        :param dps: (int) Decimal places to round the edge values. By default, this will round to 4 d.p's.\n        "
        pass

    def _get_default_stylesheet(self):
        if False:
            print('Hello World!')
        '\n        Gets the default stylesheet and adds the MST styling.\n\n        :return: (List) Returns the stylesheet to be added to the graph.\n        '
        pass