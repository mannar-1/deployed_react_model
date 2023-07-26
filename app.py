from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.utils import load_img, img_to_array
from keras.models import load_model
import keras.utils as image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from flask import request, jsonify
from flask_cors import CORS

# Define a flask app
app = Flask(__name__)
CORS(app) 

# Model saved with Keras model.save()
MODEL_PATH = 'models/food.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = load_img(img_path, target_size=(229, 229))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.  # Normalize the pixel values to [0, 1] range

    preds = model.predict(x)
    print(preds)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

classes=['Alu_Gobi', 'Alu_Matar', 'American_Pancakes', 'Apple_Pie', 'Arros_negre', 'Arroz_con_huevo', 'Arroz_con_pollo', 'Avocado_Toast', 'BLT_Sandwich', 'Bacon_Egg_and_Cheese_Sandwich', 'Bagels', 'Baguette', 'Baked_Ziti', 'Banana_Bread', 'Banana_Split', 'Barbecue_Ribs', 'Barfi', 'Beef_Rendang_(rendang)', 'Beef_Vindaloo', 'Beefaroni', 'Beignet', 'Beignets', 'Blanquette_de_Veau', 'Bocadillo_de_carne', 'Bocadillo_de_jamon', 'Bocadillo_de_pollo', 'Bocadillo_de_queso', 'Boeuf_Bourguignon', 'Bouillabaisse', 'Bread_Pudding', 'Breakfast_Burrito', 'Brioche', 'Brodetto_Di_Pesce', 'Brownies', 'Buffalo_Wings', 'Burrito', 'Butter_Chicken', 'Cabrales', 'California-Style_Pizza', 'Cannele', 'Carrot_Halwa', 'Cassoulet', 'Chaat_Papri', 'Cham-Cham', 'Chana_Masala', 'Chapati', 'Chausson_aux_Pommes', 'Cheese_Dog', 'Cheeseburger', 'Cheesesteak', 'Chiacchiere', 'Chicago-Style_Deep_Dish_Pizza', 'Chicken_65', 'Chicken_Biriyani', 'Chicken_Fried_Steak', 'Chicken_Nuggets', 'Chicken_Parmigiana', 'Chicken_Tajine', 'Chicken_Tikka', 'Chicken_and_Waffles', 'Chicken_with_Chestnuts', 'Chili_Chicken', 'Chili_Dog', 'Chili_con_Carne', 'Chimichanga', 'Chinese_Chicken_Salad', 'Chistorra', 'Chocolate_Chip_Cookie', 'Chocolate_Italian_Sponge_Cake', 'Chocolate_caliente', 'Chop_Suey', 'Chouquette', 'Club_Sandwich', 'Cobb_Salad', 'Conejo_con_arroz', 'Coq_au_Vin', 'Cordero_asado', 'Coriander_Chutney', 'Corn_on_the_Cob', 'Cornbread', 'Couscous', 'Crab_Cake', 'Creamy_Chicken_Marsala', 'Creamy_Lemon_Parmesan_Chicken_Piccata', 'Creamy_Mushroom_Soup_With_Italian_Sausage', 'Creme_Brulee', 'Crepe', 'Croissant', 'Croque_Monsieur', 'Croquetas_de_bacalao', 'Croquetas_de_jamon', 'Cuban_Sandwich', 'Cupcake', 'Daifuku', 'Daiwa_Sushi', 'Dal_Makhani', 'Dango', 'Date_Tamarind_Chutney', 'Dessert_Bars', 'Deviled_Eggs', 'Dhokla', 'Doughnut', 'Easy_Italian_Stuffed_Peppers', 'Ebi_furai', 'Eclair', 'Eggplant_And_Italian_Sausage_Gratin', 'Eggs_Benedict', 'Empanada_Galleg', 'Endo_Sushi', 'Escargot', 'Espetos', 'Fajitas', 'Fish_Fry', 'Flan', 'Focaccia_Di_Genova', 'French_Onion_Soup', 'Fried Chicken', 'Frozen_Yogurt', 'Frutti_Di_Mare', 'Fudge', 'Fuet', 'Gachas', 'Galette_Complet', 'Gambas_a_la_plancha', 'Garlic_Butter_Italian_Sausage_Sandwiches', 'Garlic_Parmesan_Cheese_Bombs', 'Garlic_Soya_Chicken', "General_Tso's_Chicken", 'Gougere', 'Gratin_Dauphinois', 'Grilled_Cheese', 'Gulab_Jamun', 'Gumbo', 'Gyoza', 'Hakata_ramen', 'Himono', 'Hoagie', 'Honey_Chilli_Potato', 'Hot_Italian_Sliders', 'Houtou_Fudou', 'Ice_Cream_Float', 'Idli', 'Italian_Beef', 'Italian_Braised_Chicken', 'Italian_Chicken_Meal_Prep_Bowls', 'Italian_Crescent_Casserole', 'Italian_Garlic_Bread_Grilled_Cheese', 'Italian_Green_Salad', 'Italian_Oven_Roasted_Vegetables', 'Italian_Ravioli', 'Italian_Roasted_Potatoes', 'Italian_Sandwich_Roll-Ups', 'Italian_Sausage_And_Peppers', 'Italian_Sausage_Rigatoni', 'Italian_Seafood_Pasta_With_Mussels_&_Calamari', 'Italian_Skillet_Chicken_With_Tomatoes_And_Mushrooms', 'Italian_Style_Chicken_Mozzarella_Skillet', 'Italian_Wedding_Soup', 'Jabugo', 'Jalebi', 'Jambalaya', 'Jambon-Beurre', 'Kaiseki', 'Kaisendon', 'Kansas_City-Style_Barbecue', 'Karaage', 'Kasutera', 'Katsukura', 'Kayu', 'Key_Lime_Pie', 'Kheema', 'Kheer', 'Korokke', 'Kulfi', 'Kung_Pao_Chicken', 'Kushiyaki', 'Ladoo', 'Lamb_Kebabs', 'Lamb_Vindaloo', 'Lime_Pickle', 'Lle_Flottante', 'Lobster_Roll', 'Longaniza', 'Mac and Cheese', 'Macaron', 'Madeleine', 'Magdalenas', 'Magret_de_canard', 'Maisen', 'Makizushi', 'Malai_Kofta_(Veggie_Balls_With_Sauce)', 'Mango_Lassi', 'Marinara_Sauce', 'Masala_Chai', 'Masala_Dosa', 'Masoor_Dal', 'Meatloaf', 'Medu_Vada', 'Milkshake', 'Mille_Feuille', 'Miso_Soup', 'Mofongo', 'Molten_Chocolate_Cake', 'Monaka', 'Montadito', 'Morcilla_de_Burgos', 'Moules_Frites', 'Mussels_In_Spicy_Red_Arrabbiata_Sauce', 'Naan', 'Napolitana_de_chocolate', 'Nashville_Hot_Chicken', 'Natillas', 'Navratan_Korma_(Nine_Gem_Curry)', 'Negima_yakitori', 'New_England_Clam_Chowder', 'New_York-Style_Cheesecake', 'New_York-Style_Pizza', 'Okonomiyaki', 'Omelette', 'Omurice', 'Onion_Pakora', 'Onion_Rings', 'Orzo_With_Italian_Sausage_And_Peppers', 'Osso_Buco', 'Oyakodon', 'Pa_amb_tomaquet', 'Pain_au_Chocolat', 'Palmie', 'Pane_Bianco', 'Papadum', 'Pasta_Salad', 'Pastrami_on_Rye', 'Pecan_Pie', 'Pernil', 'Pesto_Sliders', 'Pissaladiere', 'Pisto', "Po'Boy", 'Poke_Salad', 'Porchetta', 'Pot_Pie', 'Pot_Roast', 'Pot_au_feu', 'Pulao', 'Pulled_Pork', 'Queso_Tetilla', 'Quiche', 'Quince_Paste', 'Quinoi', 'Rabas', 'Ragu_Alla_Bolognese', 'Raita', 'Rajma', 'Ras_Malai', 'Ratatouille', 'Ravioli_With_Garlic_Basil_Oil', 'Red_Velvet_Cake', 'Reuben', 'Rillettes', 'Roast_Beef_Sandwich', 'Rogan_Josh', 'Romesco', 'Roscon_de_Reyes', "S'more", 'Saag_Paneer', 'Sambar', 'Samosa', 'Sauce_Bechamel', 'Shoyu_ramen', 'Shrimp_And_Corn_Risotto_With_Bacon', 'Shrimp_and_Grits', 'Sicilian_Pizza', 'Slow_Cooker_Italian_Beef', 'Snickerdoodle', 'Soba', 'Sofrito', 'Souffle', 'Spaghetti_Olio_E_Aglio', 'Spaghetti_and_Meatballs', 'Steak_Tartare', 'Stromboli', 'Stuffed_Italian_Flank_Steak', 'Submarine_Sandwich', 'Sukiyaki', 'Sundae', 'Surf_and_Turf', 'Tandoori_Chicken', 'Tapenade', 'Tarta_de_Santiago', 'Tarte', 'Tartine', 'Teppanyaki', 'Teriyaki', 'Texas-Style_Barbecue', 'Tomate_Farcie', 'Tomate_frito', 'Tomato_&_Basil_Bruschetta', 'Tonkotsu_ramen', 'Torta_del_Casar', 'Tortellini_Pasta_Carbonara', 'Tortellini_Soup_With_Italian_Sausage_And_Spinach', 'Tsunahachi', 'Turron', 'Udon', 'Uthapam', 'Vegetable_Jalfrezi', 'Verdejo', 'Yakisoba']
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
         
        print("ent")
        # Process your result for human
        pred_class_index = np.argmax(preds)
        print("idx",pred_class_index)
        prediction = classes[pred_class_index]
        print("enterted and returning ",prediction)
        print(prediction)
        return jsonify({'prediction': prediction})
    return None


if __name__ == '__main__':
    app.run(debug=True)