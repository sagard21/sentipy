import streamlit as st
from text_explainer import explain_text_features


# Streamlit app title
st.title('What features is your model learning?')

# Get user input
user_input = st.text_area(label="Enter your comment / tweet here")

# get number of features input
num_features_input = st.number_input(label="Num of features to visualize",
                                        min_value=1, max_value=7, step=1)

# Display outcome and pyplot graph
if user_input and num_features_input:
    outcome, graph_output = explain_text_features(user_input,
                                                    num_features_input)
    pred_class = f'## Predicted class - *{outcome}*'
    st.write(pred_class, )
    st.pyplot(graph_output)
else:
    st.write('Waiting for input')