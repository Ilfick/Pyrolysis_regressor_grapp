import gradio as gr
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train_data():
    path = 'data.csv'
    data = pd.read_csv(path, sep=',')
    X = data[['Working_time', 'Ethane', 'Prop', 'Temp', 'Vap_fr']].values
    y = data[['C2H4', 'C3H6']].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    return X


class PyrolysisPredictor():
    mdpath = 'fm_pyrolisys'
    fm_pyrolisys = load_model(mdpath)


def starting(min_working_time, max_working_time, ethane, propane, temperature, vapor):
    user_df = form_df(min_working_time, max_working_time, ethane, propane, temperature, vapor)
    user_x = user_df[['working_time', 'ethane', 'propane', 'temperature', 'vapor']].values
    sc = StandardScaler()
    sc.fit(train_data())
    ux_scaled = sc.transform(user_x)
    pred = PyrolysisPredictor.fm_pyrolisys.predict(ux_scaled)
    results = pd.DataFrame(pred)
    return results


def form_df(min_working_time, max_working_time, ethane, propane, temperature, vapor):
    df = pd.DataFrame(columns=['working_time', 'ethane', 'propane', 'temperature', 'vapor'])
    min_working_time, max_working_time = int(min_working_time), int(max_working_time)
    for i in range(min_working_time, max_working_time + 1, 8):
        data_list = [i, ethane, propane, temperature, vapor]
        df.loc[len(df)] = data_list
    return df


def res_df(min_working_time, max_working_time, ethane, propane, temperature, vapor):
    user_df = form_df(min_working_time, max_working_time, ethane, propane, temperature, vapor)
    pred_df = starting(min_working_time, max_working_time, ethane, propane, temperature, vapor)
    pred_df.columns = ['C2H4', 'C3H6']
    df_res = pd.concat([user_df, pred_df], axis=1)
    # data_csv = df_res.to_csv(os.path.join(os.path.dirname(__file__), "user_data.csv"))
    return df_res.to_csv(os.path.join(os.path.dirname(__file__), "user_data.csv"), index=False)


def make_plot(min_working_time, max_working_time, ethane, propane, temperature, vapor):
    result = starting(min_working_time, max_working_time, ethane, propane, temperature, vapor)
    fig_eth = plt.figure()
    plt.plot(result[0], label='C2H4')
    plt.title("График выхода C2H4")
    plt.ylabel("Выход")
    plt.xlabel("Пробег")
    plt.grid()
    fig_prop = plt.figure()
    plt.plot(result[1], label='C3H6')
    plt.title("График выхода C3H6")
    plt.ylabel("Выход")
    plt.xlabel("Пробег")
    plt.grid()
    return fig_eth, fig_prop


with gr.Blocks() as first_app:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                min_working_time = gr.Slider(minimum=8, maximum=1696, step=8, label="Начальное время работы печи")
                max_working_time = gr.Slider(minimum=16, maximum=1704, step=8, label="Время остановки печи")
            with gr.Row():
                ethane = gr.Slider(minimum=0, maximum=21, step=0.5, label="Доля этана")
                propane = gr.Slider(minimum=0, maximum=21, step=0.5, label="Доля пропана")
            with gr.Row():
                temperature = gr.Slider(minimum=800, maximum=900, step=1, label="Температура")
                vapor = gr.Slider(minimum=0.30, maximum=0.70, step=0.01, label="Доля пара")
            with gr.Row():
                start_btn = gr.Button(value="Запустить печь пиролиза")
                plot_btn = gr.Button(value="График выхода")
                file_btn = gr.Button(value="Сохранить данные")
        with gr.Column():
            pred_data = gr.Textbox(label="Выход: C2H4  |  C3H6")
            eth_plot = gr.Plot(label="График выхода C2H4")
            prop_plot = gr.Plot(label="График выхода C3H6")
            show_df = [gr.File(label="Output File",
                               file_count="single",
                               file_types=["", ".", ".csv", ".xls", ".xlsx"])]

    start_btn.click(starting, inputs=[min_working_time, max_working_time, ethane, propane, temperature, vapor],
                    outputs=pred_data)
    plot_btn.click(make_plot, inputs=[min_working_time, max_working_time, ethane, propane, temperature, vapor],
                   outputs=[eth_plot, prop_plot])
    file_btn.click(res_df, inputs=[min_working_time, max_working_time, ethane, propane, temperature, vapor],
                   outputs=show_df)

first_app.launch()
