import pandas as pd
import os
import plotly.graph_objects as go


# Visualization
def create_combined_parameter_sharpe_plot(results, window_index, subfolder):
    """Create and save an interactive plot of parameter combinations vs Sharpe ratio."""
    df = pd.DataFrame(results)
    df = df.sort_values('sharpe_ratio', ascending=False)
    df['param_combination'] = df['params'].apply(lambda x: ', '.join(f"{k}:{v}" for k, v in x.items()))
    df['calmar_ratio'] = df['total_return'] / df['max_drawdown'].abs()
    # Use clip(lower=0) to avoid negative total_return increasing trade_shares
    df['trade_shares'] = df['total_trades'] * df['sharpe_ratio'] * df['total_return'].clip(lower=0)
    
    fig = go.Figure()
    
    hover_text = df.apply(lambda row: f"Params: {row['param_combination']}<br>"
                                    f"Sharpe Ratio: {row['sharpe_ratio']:.4f}<br>"
                                    f"Sortino Ratio: {row['sortino_ratio']:.4f}<br>"
                                    f"Calmar Ratio: {row['calmar_ratio']:.4f}<br>"
                                    f"Total Return: {row['total_return']:.2%}<br>"
                                    f"Total Trades: {row['total_trades']}<br>"
                                    f"Trades*Sharpe*Return: {row['trade_shares']:.4f}", axis=1)
    
    scatter = go.Scatter(
        x=list(range(len(df))),
        y=df['sharpe_ratio'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['sharpe_ratio'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Sharpe Ratio')
        ),
        text=hover_text,
        hoverinfo='text'
    )
    
    fig.add_trace(scatter)
    
    for y in [-2, 0, 2]:
        fig.add_shape(
            type="line",
            x0=0,
            y0=y,
            x1=len(df) - 1,
            y1=y,
            line=dict(color="red" if y != 0 else "green", width=2, dash="dash"),
        )
    
    fig.update_layout(
        title=f'Parameter Combinations vs Sharpe Ratio (Window {window_index + 1})',
        xaxis_title='Parameter Combinations (sorted by Sharpe Ratio)',
        yaxis_title='Sharpe Ratio',
        hovermode='closest',
        height=800,
        width=1200,
    )
    
    fig.update_xaxes(showticklabels=False)
    
    os.makedirs(subfolder, exist_ok=True)
    plot_filename = os.path.join(subfolder, f'sharpe_window_{window_index}.html')
    fig.write_html(plot_filename)
    
    return plot_filename
