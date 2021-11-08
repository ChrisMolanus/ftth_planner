# Imports
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

# Page setup
# Use the full page instead of a narrow central column
# st.set_page_config(layout="wide")
sidebar_row1_col1, sidebar_row1_col2  = st.sidebar.columns(2)
#The other gadgets follow the same syntax.
sidebar_row2_col1, sidebar_row2_col2 = st.sidebar.columns(2)

sidebar_row1_col1.text_input('Top_lat', '123')
sidebar_row1_col2.text_input('Top_lon', '123')
sidebar_row2_col1.text_input('Bot_lat', '123')
sidebar_row2_col2.text_input('Bot_lon', '123')


### Header section and logo

# Write a page title
col1, col2 = st.columns((2,1))
col1.title('Fiber To The Home Network')

#Insert a picture next to title
# First, read it with PIL
image = Image.open('images/Cognizant_Logo_Blue.png')
# Load Image in the App
col2.image(image, use_column_width=True)

st.subheader('Cognizant’s fiber network optimizer \n')

### Interaction section

# Intructions
st.write('Please configure your inputs below...')

# Inputs

# Input boxes for choosing address
col1, col2 = st.columns(2)
postal_input1 = col1.number_input('Enter your postal code', 0, 4000, 0000)
postal_input2 = col2.text_input('Extension', 'AA', max_chars=2, )

# Output: data visualisation

# Map with route placeholder
st.subheader(f'Optimal fiber route to {postal_input1}{postal_input2} (place holder) \n')
image_route_map = Image.open('images/ftth_map_indexed.png')
st.image(image_route_map, use_column_width=True)

# Dataframe
st.subheader('Data data Data data... \n')
cols = list('ABCDE')
df_rand = pd.DataFrame(np.random.randint(0,100,size=(100, 5)), columns=cols)

st_ms = st.multiselect("Choose columns to display", df_rand.columns.tolist(), default=cols)
st.dataframe(df_rand[st_ms])

# TODO: Connect Dataframe to Map

############### Main.py #############################
import networkx
import osmnx as ox
import matplotlib.pyplot as plt

from cost_parameters import CostParameters
from costs import DetailedCost
from fibers import get_fiber_network
from report import get_detailed_report
from trenches2 import get_trench_network, add_trenches_to_network

def plot_network(g_box: networkx.MultiDiGraph):
    ec = ['black' if 'highway' in d else
          "grey" if "trench_crossing" in d and d["trench_crossing"]else
          "blue" if "house_trench" in d else
          'red' for _, _, _, d in g_box.edges(keys=True, data=True)]
    fig, ax = ox.plot_graph(g_box, bgcolor='white', edge_color=ec,
                            node_size=0, edge_linewidth=0.5,
                            show=False, close=False)
    ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
    plt.show()



# Get graphs of different infrastructure types, then get trenches
g_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48386, 4.49521,
                           network_type='drive',
                           simplify=True,
                           retain_all=False,
                           truncate_by_edge=True)
building_gdf = ox.geometries_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, tags={'building': True})
trench_network = get_trench_network(g_box, building_gdf)
# import pickle
# pickle.dump(trench_network, open("trench_network.p", "wb"))

trench_network_graph = add_trenches_to_network(trench_network, g_box)
ox.plot_graph(trench_network_graph)

cost_parameters = CostParameters()
fiber_network = get_fiber_network(trench_network, cost_parameters)

detailed_cost = DetailedCost(fiber_network, cost_parameters)

detailed_report = get_detailed_report(detailed_cost, building_gdf)

if detailed_report.plot is not None:
    detailed_report.plot.show()

# TODO: convert detailed_report to PDF

