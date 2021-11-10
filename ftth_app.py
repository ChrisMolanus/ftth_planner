# Imports
import streamlit as st
from PIL import Image
import osmnx as ox

from cost_parameters import CostParameters
from costs import get_costs
from fibers import get_fiber_network
from trenches2 import get_trench_network, add_trenches_to_network

def get_planning():
    box = (float(north), float(south), float(east), float(west))
    g_box = ox.graph_from_bbox(*box,
                               network_type='drive',
                               simplify=False,
                               retain_all=False,
                               truncate_by_edge=True)
    building_gdf = ox.geometries_from_bbox(*box, tags={'building': True})
    trench_network = get_trench_network(g_box, building_gdf)
    trench_network_graph = add_trenches_to_network(trench_network, g_box)
    cost_parameters = CostParameters()
    fiber_network, fig = get_fiber_network(trench_network, cost_parameters, building_gdf, g_box)
    detailed_cost = get_costs(fiber_network, cost_parameters)
    return detailed_cost, fig

# Sidebar with coordinate/placename inputs
north_field, south_field = st.sidebar.columns(2)
east_field, west_field = st.sidebar.columns(2)

# Write a page title
col1, col2 = st.columns((2, 1))
col1.title('Fiber To The Home Network')

#Insert a picture next to title
image = Image.open('images/Cognizant_Logo_Blue.png')
col2.image(image, use_column_width=True)

st.subheader('Cognizant’s fiber network optimizer \n')


# Sidebar inputs
north = north_field.text_input('North','50.78694')
south = south_field.text_input('South','50.77902')
east = east_field.text_input('East', '4.48386')
west = west_field.text_input('West', '4.49521')

detailed_cost, fig = get_planning()

# Map with optimal fiber route
st.subheader(f'Optimal fiber network route for [N:{north}, S:{south}, E:{east}, W:{west}] \n')
st.pyplot(fig)

st.header('Cost data \n')
st.subheader('Material Costs \n')
materials_df = detailed_cost.get_materials_dataframe()
materials_df.set_index('Type', inplace=True)
st.dataframe(materials_df)

st.subheader('Labour Costs \n')
labor_df = detailed_cost.get_labor_dataframe()
labor_df.set_index('Type', inplace=True)
st.dataframe(labor_df)


