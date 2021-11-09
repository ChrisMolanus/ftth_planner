# Imports
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import networkx
import osmnx as ox
import matplotlib.pyplot as plt

from cost_parameters import CostParameters
from costs import DetailedCost, get_costs
from fibers import get_fiber_network, EquipmentType, plot_fiber_network
from report import get_detailed_report
from trenches2 import get_trench_network, add_trenches_to_network

#### Page setup

# Sidebar with coordinate/placename inputs
north_field, south_field = st.sidebar.beta_columns(2)
east_field, west_field = st.sidebar.beta_columns(2)

# 50.843217, 50.833949, 4.439903, 4.461962
# 1,3 51.98446, 5.64113
# 2,4 51.98, 5.6575
# Blijdorp
# 51.9317, 4.4544
# 51.9292, 4.4637
#### Header section and logo

# Write a page title
col1, col2 = st.beta_columns((2, 1))
col1.title('Fiber To The Home Network')

#Insert a picture next to title
# First, read it with PIL
image = Image.open('images/Cognizant_Logo_Blue.png')
# Load Image in the App
col2.image(image, use_column_width=True)

st.subheader('Cognizant’s fiber network optimizer \n')

### Interaction section

# Intructions
# st.write('Please configure your inputs below...')

# Inputs

# Sidebar inputs
north = north_field.text_input('North','50.78694')
south = south_field.text_input('South','50.77902')
east = east_field.text_input('East', '4.48386')
west = west_field.text_input('West', '4.49521')

# Input boxes for choosing address
# col1, col2 = st.columns(2)
# postal_input1 = col1.number_input('Enter your postal code', 0, 4000, 0000)
# postal_input2 = col2.text_input('Extension', 'AA', max_chars=2, )

# Output: data visualisation



# TODO: Connect Dataframe to Map

############### Main.py #############################

def plot_network(g_box: networkx.MultiDiGraph):
    ec = ['black' if 'highway' in d else
          "grey" if "trench_crossing" in d and d["trench_crossing"]else
          "blue" if "house_trench" in d else
          'red' for _, _, _, d in g_box.edges(keys=True, data=True)]
    fig, ax = ox.plot_graph(g_box, bgcolor='white', edge_color=ec,
                            node_size=0, edge_linewidth=0.5,
                            show=False, close=False)
    ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
    return(fig)


# Get graphs of different infrastructure types, then get trenches
box = (float(north), float(south), float(east), float(west))
g_box = ox.graph_from_bbox(*box,
                           network_type='drive',
                           simplify=False,
                           retain_all=False,
                           truncate_by_edge=True)
building_gdf = ox.geometries_from_bbox(*box, tags={'building': True})
trench_network = get_trench_network(g_box, building_gdf)
#import pickle
#pickle.dump(trench_network, open("trench_network.p", "wb"))

trench_network_graph = add_trenches_to_network(trench_network, g_box)


cost_parameters = CostParameters()
fiber_network, fig = get_fiber_network(trench_network, cost_parameters, building_gdf, g_box)

detailed_cost = get_costs(fiber_network, cost_parameters)

# TODO: convert detailed_report to PDF

#####################################################

# Map with optimal fiber route
st.subheader(f'Optimal fiber network route for [N:{north}, S:{south}, E:{east}, W:{west}] \n')
# image_route_map = Image.open('images/ftth_map_indexed.png')
# st.image(image_route_map, use_column_width=True)
st.pyplot(fig)

st.header('Cost data \n')
# Subheader for cost data


# Dataframe
#cols_field, data_frame = st.beta_columns([1, 3])

#cols = ["Type", "Quantity", "Quantity units", "Total Cost"]
st.subheader('Material data \n')
materials_df = detailed_cost.get_materials_dataframe()
#materials_selected_cols = cols_field.multiselect("Costs for Materials", materials_df.columns.tolist(), default=cols)
st.dataframe(materials_df)

st.subheader('Labour data \n')
labor_df = detailed_cost.get_labor_dataframe()
#labor_selected_cols = cols_field.multiselect("Costs for Materials", materials_df.columns.tolist(), default=cols)
st.dataframe(labor_df)


