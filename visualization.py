import plotly.express as px

def scatter_sp_attack_defense(df):
    fig = px.scatter(
        df, 
        x='sp_attack', 
        y='sp_defense',
        title="Scatter Plot: SP Attack vs SP Defense",
        color='sp_attack',
        opacity=0.6
    )
    fig.show()

def histogram_against_columns(df, against_cols):
    plot_df = df[against_cols].melt(var_name='Type', value_name='Value')
    fig = px.histogram(
        plot_df,
        x='Type',
        color='Value',
        barmode='group',
        title="Frequency of 0/1 for 'Against' Columns",
        height=500,
        width=1000
    )
    fig.show()
    
def scatter_3d(df, x_col, y_col, z_col, color_col=None, size_col=None, title=None):
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        size=size_col,
        hover_data=[x_col, y_col, z_col],
        opacity=0.7,
        title=title,
        width=1200,
        height=600
    )
    fig.show()
