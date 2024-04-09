import os
import cv2
import numpy as np
import tensorflow as tf
import telebot
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1
    img = np.expand_dims(img, 0)
    img = tf.convert_to_tensor(img)
    return img

def preprocess_image(img, target_dim=224):
    shape = tf.cast(tf.shape(img)[1:-1], tf.float32)
    min_side = min(shape)
    scale = target_dim / min_side
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = tf.image.resize_with_crop_or_pad(img, target_dim, target_dim)
    return img

def cartoonify_image(img_path):
    input_img = load_image(img_path)
    preprocessed_img = preprocess_image(input_img, target_dim=512)

    model_path = '1.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
    interpreter.invoke()

    raw_output = interpreter.tensor(interpreter.get_output_details()[0]['index'])()

    output_img = (np.squeeze(raw_output) + 1.0) * 127.5
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    return output_img

bot = telebot.TeleBot('6711556035:AAFquCQIUueoZOwziS9znkqRudFAIrmT3HY')

if not os.path.exists('input_images'):
    os.makedirs('input_images')

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "Привет":
        bot.send_message(message.from_user.id, "Привет")
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Отправь фото")
    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    photo = message.photo[-1]
    file_id = photo.file_id
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path

    downloaded_file = bot.download_file(file_path)

    with open(f'input_images/{file_id}.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)

    processed_img = cartoonify_image(f'input_images/{file_id}.jpg')

    output_path = f"output_images/{file_id}.jpg"
    cv2.imwrite(output_path, processed_img)

    with open(output_path, 'rb') as output_file:
        bot.send_photo(message.chat.id, output_file)

    bot.send_message(message.chat.id, "Держи")

bot.polling(none_stop=True, interval=0)
