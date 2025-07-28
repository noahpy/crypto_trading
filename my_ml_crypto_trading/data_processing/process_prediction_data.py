import json
import plotly.graph_objects as go

def plot_predicted_vs_actual_midprice_interactive(file_path):
    timestamps = []
    predicted_midprices = []
    actual_midprices = []

    horizon_offset = 10

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            timestamps.append(data["timestamp"])
            predicted = data["midprice_before"] + data["prediction_before"]
            actual = data["midprice_current"]

            predicted_midprices.append(predicted)
            actual_midprices.append(actual)

    # Convert timestamps to seconds relative to first timestamp
    t0 = timestamps[0]
    time_seconds = [(t - t0) / 1000 for t in timestamps]

    if horizon_offset > 0:
        time_seconds = [0 for _ in range(horizon_offset)] + time_seconds
        predicted_midprices = predicted_midprices + [predicted_midprices[-1] for _ in range(horizon_offset)] 
        actual_midprices = [actual_midprices[0] for _ in range(horizon_offset)] + actual_midprices

    # Create interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_seconds, y=predicted_midprices, mode='lines', name='Predicted Midprice', line=dict(dash='dash', color='orange')))
    fig.add_trace(go.Scatter(x=time_seconds, y=actual_midprices, mode='lines', name='Actual Midprice', line=dict(color='blue')))

    fig.update_layout(
        title='Predicted vs Actual Midprice Over Time',
        xaxis_title='Time (s)',
        yaxis_title='Midprice',
        legend=dict(x=0, y=1),
        hovermode='x unified',
        template='plotly_white'
    )

    fig.show()

# Example usage
# Replace with your actual file path
plot_predicted_vs_actual_midprice_interactive("prediction.data")
