# Imports
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import osmnx as ox

from cost_parameters import CostParameters
from costs import get_costs
from fibers import get_fiber_network
from trenches2 import get_trench_network, get_trench_to_network_graph


def get_planning():
    box = (float(north), float(south), float(east), float(west))
    ref_g_box = ox.graph_from_bbox(*box,
                               network_type='drive',
                               simplify=False,
                               retain_all=False,
                               truncate_by_edge=True)
    ref_building_gdf = ox.geometries_from_bbox(*box, tags={'building': True})
    return ref_g_box, ref_building_gdf


def get_trench_graph(ref_g_box, ref_building_gdf):
    ref_trench_network = get_trench_network(ref_g_box, ref_building_gdf)
    ref_trench_network_graph = get_trench_to_network_graph(ref_trench_network, ref_g_box)
    return ref_trench_network, ref_trench_network_graph


def get_fiber_planning(ref_trench_network, ref_building_gdf, ref_g_box):
    cost_parameters = CostParameters()
    fiber_network, ref_fig = get_fiber_network(ref_trench_network, cost_parameters, ref_building_gdf, ref_g_box)
    ref_detailed_cost = get_costs(fiber_network, cost_parameters)
    return ref_detailed_cost, ref_fig


def plot_1(ref_g_box, ref_building_gdf, trench_graph):
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
    ox.plot_footprints(building_gdf, ax=ax, color="burlywood", alpha=0.6, show=False, close=False)
    return ref_fig


# Sidebar with coordinate/placename inputs
st.sidebar.subheader('Input Coordinates')
# north_field, south_field = st.sidebar.columns(2)
# east_field, west_field = st.sidebar.columns(2)

#box_field = st.sidebar.columns(1)
box_text = st.sidebar.text_input('North, East, South, West', '51.1771, 4.4057, 51.1754, 4.4121')

# Write a page title
col1, col2 = st.columns((2, 1))
col1.title('Fiber To The Home Network')

# Insert a picture next to title
image = Image.open('images/Cognizant_Logo_Blue.png')
col2.image(image, use_column_width=True)

st.subheader('Cognizant’s fiber network optimizer \n')


# Sidebar inputs
# north = north_field.text_input('North', '50.78694')
# south = south_field.text_input('South', '50.77902')
# east = east_field.text_input('East', '4.48386')
# west = west_field.text_input('West', '4.49521')

north, east, south, west = str(box_text).split(",")

# Map
st.sidebar.subheader('Map')
map_data = {'lat': [np.average([float(north), float(south)])], 'lon': [np.average([float(east), float(west)])]}
map_df = pd.DataFrame(data=map_data)
st.sidebar.map(map_df, zoom=8)


# Map with optimal fiber route
st.subheader(f'Optimal fiber network route for [N:{north}, S:{south}, E:{east}, W:{west}] \n')

g_box, building_gdf = get_planning()
plot_holder = st.empty()
plot_holder.pyplot(plot_1(g_box, building_gdf, None))

trench_network, trench_network_graph = get_trench_graph(g_box, building_gdf)
plot_holder.pyplot(plot_1(g_box, building_gdf, trench_network_graph))

detailed_cost, fig = get_fiber_planning(trench_network, building_gdf, g_box )
fig.legend(loc='lower center', fontsize='x-small')
plot_holder.pyplot(fig)


# cost dataframes
materials_df = detailed_cost.get_materials_dataframe()
materials_df.set_index('Type', inplace=True)
materials_df["Quantity"] = materials_df["Quantity"].round(decimals=2)
materials_total = materials_df['Total Cost'].sum()
# materials_df.loc[['Total'], ['Quantity', 'Quantity units']] = "-"


labor_df = detailed_cost.get_labor_dataframe()
labor_df.set_index('Type', inplace=True)
labor_df["Quantity"] = labor_df["Quantity"].round(decimals=2)
labor_total = labor_df['Total Cost'].sum()
# labor_df.replace('NA', '', regex=False, inplace=True)


# Display dataframes
st.header('Material cost breakdown \n')
cols_materials = list(materials_df.columns.values)
ms_mat = st.multiselect("Select dataframe columns", materials_df.columns.tolist(), default=cols_materials, key=1)
st.dataframe(materials_df[ms_mat].style.set_precision(2))

_, materials_total_col = st.columns([3, 1])
materials_total_col.subheader("€{:,.2f}".format(materials_total))

st.header('Labour cost breakdown \n')
cols_labor = list(labor_df.columns.values)
ms_lab = st.multiselect("Select dataframe columns", labor_df.columns.tolist(), default=cols_labor, key=2)
st.dataframe(labor_df[ms_lab].style.set_precision(2))

_, labor_total_col = st.columns([3, 1])
labor_total_col.subheader("€{:,.2f}".format(labor_total))
