# Imports
from typing import Tuple, Optional

import geopandas
import matplotlib.pyplot as plt
import networkx
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
import osmnx as ox

from cost_parameters import CostParameters
from costs import get_costs, DetailedCost
from fibers import get_fiber_network
from trenches import get_trench_network, get_trench_to_network_graph, TrenchNetwork


def get_planning(north: str, south: str, east: str, west: str) -> Tuple[networkx.MultiDiGraph, geopandas.GeoDataFrame]:
    """
    Get open Street map data, the g_box graph and a GeoDataFrame with the building data
    :param north: Upper left GPS Latitude of box
    :param south: Upper left GPS Longitude of box
    :param east: Lower left GPS Latitude of box
    :param west: Lower left GPS Longitude of box
    :return: g_box, building_gdf
    """
    box = (float(north), float(south), float(east), float(west))
    ref_g_box = ox.graph_from_bbox(*box,
                               network_type='drive',
                               simplify=False,
                               retain_all=False,
                               truncate_by_edge=True)
    ref_building_gdf = ox.geometries_from_bbox(*box, tags={'building': True})
    return ref_g_box, ref_building_gdf


def get_trench_graph(ref_g_box: networkx.MultiDiGraph,
                     ref_building_gdf: geopandas.GeoDataFrame) -> Tuple[TrenchNetwork, networkx.MultiDiGraph]:
    """
    Get trench network
    :param ref_g_box: The Open street maps road graph
    :param ref_building_gdf: The Open street maps GeoPandas Data Frame of the building data
    :return: trench_network, trench_network_graph
    """
    ref_trench_network = get_trench_network(ref_g_box, ref_building_gdf)
    ref_trench_network_graph = get_trench_to_network_graph(ref_trench_network, ref_g_box)
    return ref_trench_network, ref_trench_network_graph


def get_fiber_planning(ref_trench_network: TrenchNetwork,
                       ref_building_gdf: geopandas.GeoDataFrame,
                       ref_g_box: networkx.MultiDiGraph) -> Tuple[DetailedCost, plt.Figure]:
    """
    Get the detailed cost breakdown and a plot of the fiber network
    :param ref_trench_network: The Trench Network object
    :param ref_building_gdf: The open street map building data
    :param ref_g_box: The open street map road graph
    :return: detailed_cost, fig
    """
    cost_parameters = CostParameters()
    fiber_network, ref_fig = get_fiber_network(ref_trench_network, cost_parameters, ref_building_gdf, ref_g_box)
    ref_detailed_cost = get_costs(fiber_network, cost_parameters)
    return ref_detailed_cost, ref_fig


def plot_graph(ref_g_box: networkx.MultiDiGraph,
               ref_building_gdf: geopandas.GeoDataFrame,
               trench_graph: Optional[networkx.MultiDiGraph]) -> plt.Figure:
    """
    Returns a figure of the plot containing the roads, buildings, and trenches if available
    :param ref_g_box: The open street maps road graph
    :param ref_building_gdf: The open street maps building data
    :param trench_graph: (Optional) the trench graph
    :return: figure
    """
    ec = ['black' if 'highway' in d else
          'red' for _, _, _, d in ref_g_box.edges(keys=True, data=True)]
    ref_fig, ax = ox.plot_graph(ref_g_box, bgcolor='white', edge_color=ec,
                                node_size=0, edge_linewidth=0.6,
                                show=False, close=False)
    if trench_graph is not None:
        ec = ["grey" if "trench_crossing" in d and d["trench_crossing"] else
              "blue" if "house_trench" in d else
              'red' for _, _, _, d in trench_graph.edges(keys=True, data=True)]
        ref_fig, ax = ox.plot_graph(trench_graph, bgcolor='white', edge_color=ec,
                                    node_size=0, edge_linewidth=0.6,
                                    show=False, close=False, ax=ax)
    ox.plot_footprints(ref_building_gdf, ax=ax, color="burlywood", alpha=0.6, show=False, close=False)
    return ref_fig


# Sidebar with coordinate/placename inputs
st.sidebar.subheader('GPS Box:')
box_text = st.sidebar.text_input('Top Left (Lat,Long) , Bottom Right (Lat,Long)', '50.78694, 4.48386, 50.77902, 4.49521')

north, east, south, west = str(box_text).split(",")
st.sidebar.write("Examples:")
st.sidebar.write("50.9355,4.4853,50.9225,4.5005")
st.sidebar.write("51.1948,4.5519,51.18,4.5728")
st.sidebar.write("50.843217,4.439903,50.833949,4.461962")

# Map
try:
    st.sidebar.subheader('Map')
    map_data = {'lat': [np.average([float(north), float(south)])], 'lon': [np.average([float(east), float(west)])]}
    map_df = pd.DataFrame(data=map_data)
    st.sidebar.map(map_df, zoom=8)
except requests.exceptions.ConnectionError:
    st.write("Map could not be loaded")

# Write a page title
col1, col2 = st.columns((2, 1))
col1.title('Fiber To The Home Network planner')
col1.title("Proof of concept")
st.write("This is a Proof of concept, and not intended to be highly accurate.")

# Insert a picture next to title
image = Image.open('images/Cognizant_Logo_Blue.png')
col2.image(image, use_column_width=True)

# Map with optimal fiber route
st.subheader(f'PON planning \n')

# Progressively build up map by swapping in and out the plots as they are created
# First plot the road network
g_box, building_gdf = get_planning(north, south, east, west)
plot_holder = st.empty()
plot_holder.pyplot(plot_graph(g_box, building_gdf, None))

# Then plot the trench network
trench_network, trench_network_graph = get_trench_graph(g_box, building_gdf)
plot_holder.pyplot(plot_graph(g_box, building_gdf, trench_network_graph))

# Then if we have enough building data plot the fiber network with the detailed costs
number_of_missing_addresses = building_gdf['addr:street'].isna().sum()
number_of_addresses = len(building_gdf)-number_of_missing_addresses
if 96 < number_of_addresses < 2000:
    detailed_cost, fig = get_fiber_planning(trench_network, building_gdf, g_box )
    fig.legend(loc='lower center', fontsize='x-small')
    plot_holder.pyplot(fig)

    format_mapping = {"Quantity": "{:,.0f}", "Quantity units": "", "Total Cost": "€{:,.2f}"}

    # Material cost dataframes
    materials_df = detailed_cost.get_materials_dataframe()
    materials_df.set_index('Type', inplace=True)
    materials_df["Quantity"] = materials_df["Quantity"].round(decimals=2)
    materials_total = materials_df['Total Cost'].sum()

    # Display dataframe
    st.header('Material cost breakdown \n')
    cols_materials = list(materials_df.columns.values)
    ms_mat = st.multiselect("Select dataframe columns", materials_df.columns.tolist(), default=cols_materials, key=1)
    st.dataframe(materials_df[ms_mat].style.format(format_mapping))

    # Display sum of material cost
    _, materials_total_col = st.columns([3, 1])
    materials_total_col.subheader("€{:,.2f}".format(materials_total))

    # Labor cost dataframes
    labor_df = detailed_cost.get_labor_dataframe()
    labor_df.set_index('Type', inplace=True)
    labor_df["Quantity"] = labor_df["Quantity"].round(decimals=2)
    labor_total = labor_df['Total Cost'].sum()

    # Display dataframe
    st.header('Labour cost breakdown \n')
    cols_labor = list(labor_df.columns.values)
    ms_lab = st.multiselect("Select dataframe columns", labor_df.columns.tolist(), default=cols_labor, key=2)
    st.dataframe(labor_df[ms_lab].style.format(format_mapping))

    # Display sum of labor cost
    _, labor_total_col = st.columns([3, 1])
    labor_total_col.subheader("€{:,.2f}".format(labor_total))
else:
    if number_of_addresses < 96:
        st.write("Too many missing address data, or not enough buildings in area")
    else:
        st.write("The area is to large, and it would take to long to plan")
