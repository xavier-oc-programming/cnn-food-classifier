import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('food_classifier.h5')
IMG_SIZE = 224

CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio',
    'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding',
    'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli',
    'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake',
    'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings',
    'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame',
    'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame',
    'eggs_benedict', 'escargots', 'falafel', 'filet_mignon',
    'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup',
    'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
    'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese',
    'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette',
    'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes',
    'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop',
    'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen',
    'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
    'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]


def predict(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    top3_indices = predictions.argsort()[-3:][::-1]
    return {
        CLASSES[i].replace('_', ' ').title(): float(predictions[i])
        for i in top3_indices
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Upload a food photo'),
    outputs=gr.Label(num_top_classes=3, label='Top 3 Predictions'),
    title='Food Image Classifier — 101 Categories',
    description=(
        'Upload any food photo and the model identifies it from 101 categories. '
        'Built with MobileNetV2 transfer learning trained on the Food-101 dataset. '
        'Shows top 3 predictions with confidence scores.'
    ),
    examples=[],
    theme=gr.themes.Soft()
)

if __name__ == '__main__':
    demo.launch()
