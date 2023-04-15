import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lime.lime_text import LimeTextExplainer
class_names = ['Negative', 'Positive']
exp = [('fake', -0.35868796602489883),
 ('cheap', 0.11315675672659964),
 ('Beware', 0.028218259976994325),
 ('Let', -0.026442437335931917),
 ('to', 0.02544148766195618),
 ('counterfeits', 0.021954612619302635)]
df_weights = pd.DataFrame(exp, columns=['Words', 'Weights'])
# Convert weights to absolute value
df_weights['Weights_abs'] = df_weights['Weights'].abs()
# Round off weights to 2dp
df_weights['Weights'] = df_weights['Weights'].round(2)
# Differentiate positives and negatives 
df_weights['Color'] = np.where(df_weights['Weights'] < 0, 'red', 'green')
# Model's prediction probabilities
fig = px.bar(df_weights, x='Weights', y='Words', text_auto=True,
            #  facet_col='Weights_abs',  # Use 'Category' column for faceting
             category_orders={'Words': df_weights.sort_values('Weights_abs', ascending=False)['Words'].values})
fig.update_traces(marker_color=df_weights['Color'])
fig.update_layout(
    title = {
        'text': f"<span style='color:red'>{class_names[0]}</span> vs <span style='color:green'>{class_names[1]}</span>",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    font = {
        'size': 20
    }
)
# print(df_weights)
# # Create bar trace
# trace = go.Bar(
# x=df_weights['Weights'], 
# y=df_weights['Words'],
# # marker=dict(
# #     color=df_weights['Color']
# # )
# )

# # Create layout
# layout = go.Layout(
# title=f'{class_names[0]}\t{class_names[1]}',
# )

# # Create figure
# fig = go.Figure(data=[trace], layout=layout)

# Show plot
fig.write_image('./reports/lime_text.png')