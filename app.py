import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import base64
import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Paths and model setup
model_path = '/Users/amandanassar/Desktop/V60 NOK AI/my_model.h5'
data_dir = '/Users/amandanassar/Desktop/V60 NOK AI/data/Data images'
output_dirs = {
    'camerafault': '/Users/amandanassar/Desktop/V60 NOK AI/data/camerafault',
    'realNOK': '/Users/amandanassar/Desktop/V60 NOK AI/data/realNOK',
    'external': '/Users/amandanassar/Desktop/V60 NOK AI/data/external'
}

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "V60 Image Classification"

# Layout of the app with tabs
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('V60 Image Classification', className='text-center text-primary'), width=12)
    ], className='mb-4'),
    
    dbc.Tabs([
        dbc.Tab(label='Upload Images', children=[
            dbc.Row([
                dbc.Col(html.H3('Upload Images to Classify', className='text-center text-info'), width=12)
            ], className='mb-4'),
            
            dbc.Row([
                dbc.Col(dcc.Upload(
                    id='upload-data',
                    children=dbc.Button('Upload Images', color='success', className='mr-2'),
                    multiple=True
                ), width=12)
            ], className='mb-4'),
            
            dbc.Row([
                dbc.Col(html.Div(id='output-data-upload'), width=12)
            ], className='mb-4')
        ]),
        
        dbc.Tab(label='Run Process', children=[
            dbc.Row([
                dbc.Col(html.Div(id='image-display', style={'height': '400px', 'overflowY': 'scroll', 'border': '1px solid #39ff14'}), width=12)
            ], className='mb-4'),
            
            dbc.Row([
                dbc.Col(dbc.Button('Run Process', id='process-button', n_clicks=0, color='success'), width=12)
            ], className='mb-4'),
            
            dbc.Row([
                dbc.Col(dbc.Progress(id='progress-bar', value=0, max=100, className='mb-4'), width=12)
            ], className='mb-4'),
            
            dbc.Row([
                dbc.Col(html.Div(id='process-output'), width=12)
            ], className='mb-4')
        ])
    ])
], fluid=True)

# Helper function to decode and process images
def process_image(image_contents):
    try:
        # Decode image
        image_data = base64.b64decode(image_contents.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB').resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return image, img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

@app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents')]
)
def update_output(uploaded_files):
    if uploaded_files:
        children = []
        for index, file in enumerate(uploaded_files):
            content_type, content_string = file.split(',')
            
            # Determine file extension based on content type
            if 'image/jpeg' in content_type:
                file_extension = 'jpg'
            elif 'image/png' in content_type:
                file_extension = 'png'
            elif 'image/bmp' in content_type:
                file_extension = 'bmp'
            else:
                file_extension = 'png'  # Default extension if not recognized

            file_name = f'image_{index}.{file_extension}'

            children.append(dbc.Card([
                dbc.CardBody([
                    html.H5(file_name),
                    html.Img(src=file, style={'height': '100px', 'width': 'auto'})
                ])
            ], className='mb-2'))
        return children
    return 'No images uploaded.'

@app.callback(
    [Output('process-output', 'children'),
     Output('image-display', 'children'),
     Output('progress-bar', 'value')],
    [Input('process-button', 'n_clicks')],
    [State('upload-data', 'contents')]
)
def process_images(n_clicks, uploaded_files):
    if n_clicks > 0 and uploaded_files:
        if model is None:
            return "Model not loaded.", '', 0

        log = []
        images = []
        num_images = len(uploaded_files)
        processed_count = 0

        for index, file in enumerate(uploaded_files):
            img, img_array = process_image(file)
            if img_array is None:
                log.append(f"Error processing image.")
                continue

            prediction = model.predict(img_array)
            class_index = np.argmax(prediction[0])
            reversed_class_names = ['camerafault', 'external', 'realNOK']
            class_name = reversed_class_names[class_index]
            if not os.path.exists(output_dirs[class_name]):
                os.makedirs(output_dirs[class_name])
            
            # Save processed image with original file name and extension
            content_type, content_string = file.split(',')
            file_extension = 'png'
            if 'image/jpeg' in content_type:
                file_extension = 'jpeg'
            elif 'image/png' in content_type:
                file_extension = 'png'
            elif 'image/bmp' in content_type:
                file_extension = 'bmp'

            image_name = f'processed_image_{index}.{file_extension}'
            img.save(os.path.join(output_dirs[class_name], image_name), format=file_extension.upper())

            log.append(f"Processed image saved to {class_name} as {image_name}.")
            images.append(html.Img(src=file, style={'height': '200px', 'width': 'auto', 'margin': '5px'}))
            processed_count += 1
            progress = int((processed_count / num_images) * 100)
        
        return html.Pre("\n".join(log), style={'color': '#39ff14'}), html.Div(images, style={'display': 'flex', 'flexWrap': 'wrap'}), progress
    return '', '', 0

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
