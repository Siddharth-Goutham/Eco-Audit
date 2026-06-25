import time
import streamlit as st
from main import *

st.set_page_config(page_title="EcoAudit AI", page_icon="🌍", layout="centered")

st.title("EcoAudit AI")
st.markdown("Your localized, multi-agent sustainability advisor.")

with st.sidebar:
    st.header("Your Profile")
    name = st.text_input("Name")
    location = st.text_input("City")
    region = st.text_input("Region")
    housing = st.selectbox("Housing Status", ["Renter", "Owner"])
    distance = st.number_input("Distance travelled per day (Km)")
    commute = st.text_input("Primary Commute")


if st.button("Generate Action Plan"):
    initial_state = {
        'location': location,
        'commute_method': commute,
        'distance': distance,
        'region': region,
        'housing_status': housing,
        'username': name
    }

    config = {"configurable": {"thread_id": "session_1"}}

    status_container = st.empty()
    with st.status("Running analysis...", expanded=True) as status:
        st.write("Invoking the graph...")
        time.sleep(5)
        st.write("Interacting with agents...")
        time.sleep(10)
        st.write("Generating Report...")

        final_state = graph.invoke(initial_state)

        status.update(label="Report generated successfully!", state="complete", expanded=False)

    address = final_state['summary_dict'][2]['E-Waste center']
    st.header("E-Waste center near you")
    st.markdown(address)

    govt_subs= final_state['summary_dict'][1]['govt_subsidy']
    st.header("Government Schemes")
    st.markdown(govt_subs)

    report_text = final_state['summary_dict'][-1]['Final_Report']
    st.markdown(report_text)