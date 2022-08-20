import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import os
import random
from tensorflow.keras.models import load_model
model = keras.models.load_model('animal.hdf5')
classes = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
translate = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]


st.title("Kwara State University, Malete")
st.subheader('Animal Image Classifier - Data-set: Animal-10 from kaggle')


image=Image.open('logo.png')
st.sidebar.image(image)
st.markdown("<h1 style='text-align: center; color:#7a000d;'>Animal Identification</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center; color:#7a000d;'>Find out your perfect customer.</h3>",unsafe_allow_html=True)



upload = st.sidebar.file_uploader(label='Upload the Image')

def info(prediction):
    if prediction == 'dog':
        st.write('Dogs (Canis lupus familiaris) are domesticated mammals, not natural wild animals. They were originally bred from wolves. They have been bred by humans for a long time, and were the first animals ever to be domesticated. There are different studies that suggest that this happened between 15.000 and 100.000 years before our time. The dingo is also a dog, but many dingos have become wild animals again and live independently of humans in the range where they occur (parts of Australia)')
    elif prediction == 'horse':
        st.write('Horses are mammals of the family Equidae. They are herbivores, which means they                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   eat grass and other plants. Some plants are dangerous for them like ragwort, lemongrass (oil grass) and sometimes acorns. The common horse is the species Equus caballus. It was domesticated from wild horses by humans at least 5000 years ago. They are large, strong animals, and some breeds are used to pull heavy loads. Racehorses can gallop up to 30 miles an hour. A male horse is a stallion, and a female horse is a mare. The general term for a young horse is foal. A young female horse is a filly, and a young male horse is a colt. A castrated horse is a gelding. Horses have hooves which need protection by horseshoes from hard or rough ground.')
    elif prediction == 'elephant':
        st.write('Elephants are the largest living land mammals. The largest elephant recorded was one shot in Angola, 1974. It weighed 27,060 pounds (12.25 tonnes) and stood 13 feet 8 inches (2.16 m) tall. Their skin color is grey. At birth, an elephant calf may weigh as much as 100 kg (225 pounds). The baby elephant develops for 20 to 22 months inside its mother. No other land animal takes this long to develop before being born. In the wild, elephants have strong family relationship. Their ways of acting toward other elephants are hard for people to understand. They talk to each other with very low sounds. Most elephants sounds are so low, people cannot hear them. But elephants can hear these sounds far away. Elephants have strong, leathery skin to protect themselves.')
    elif prediction == 'butterfly':                                         
        st.write('A butterfly is a usually day-flying insect of the order Lepidoptera. They are grouped together in the suborder Rhopalocera. Butterflies are closely related to moths, from which they evolved. The earliest discovered fossil moth dates to 200 million years ago. The life of butterflies is closely connected to flowering plants, which their larvae (caterpillars) feed on, and their adults feed and lay their eggs on. They have a long-lasting history of co-evolution with flowering plants. Many of the details of plant anatomy are related to their pollinators, and vice versa. The other notable features of butterflies are their extraordinary range of colours and patterns, and their wings. These are discussed below.')
    elif prediction == 'chicken':
        st.write('A chicken (Gallus gallus domesticus) is a kind of domesticated bird. It is raised in many places for its meat and eggs. They are usually kept by humans as livestock. Most breeds of chickens can fly for a short distance. Some sleep in trees (if there are trees around). A male chicken is called a rooster or a cock(erel). A female chicken is called a hen; a young chicken is called a chick. Like other female birds, hens lay eggs. The eggs hatch into chicks. When raising chickens, a farmer needs a chicken coop (like a little house) for the chickens to roost (sleep) in. They also need a run or yard where they can exercise, take dust baths, eat and drink. The chickens also need to be protected from predators such as foxes. Fences are often used for this. Chickens can also be farmed intensively. This lets farms make a lot of chicken meat and eggs.')
    elif prediction == 'cat':
        st.write('Cats, also called domestic cats (Felis catus), are small, carnivorous mammals, of the family Felidae. Domestic cats are often called house cats when kept as indoor pets. Cats have been domesticated (tamed) for nearly 10,000 years. They are one of the most popular pets in the world. They are kept by humans for hunting rodents and as companions. There are also farm cats, which are kept on farms to keep rodents away; and feral cats, which are domestic cats that live away from humans. A cat is sometimes called a kitty. A young cat is called a kitten. A female cat is called a queen. A male cat is called a tom. There are about 60 breeds of cat. Domestic cats are found in shorthair, longhair, and hairless breeds. Cats which are not specific breeds can be referred to as domestic shorthair (DSH) or domestic longhair (DLH). The word cat is also used for other felines. Felines are usually called either big cats or small cats. The big, wild cats are well known: lions, tigers, leopards, jaguars, pumas, and cheetahs. There are small, wild cats in most parts of the world, such as the lynx in northern Europe. The big cats and wild cats are not tame, and can be very dangerous.')
    elif prediction == 'cow':
        st.write('Cattle is a word for certain mammals that belong to the genus Bos. Cattle may be cows, bulls, oxen, or calves. Cattle are the most common type of large domesticated hoofed animals. They are a prominent modern member of the subfamily Bovinae. Cattle are large grazing animals with two-toed or cloven hooves and a four-chambered stomach. This stomach is an adaptation to help digest tough grasses. Cattle can be horned or polled (or hornless), depending on the breed. The horns come out on either side of the head above the ears and are a simple shape, usually curved upwards but sometimes down. Cattle usually stay together in groups called herds. One male, called a bull will usually have a number of cows in a herd as his harem. The cows usually give birth to one calf a year, though twins are also known to be born. The calves have long strong legs and can walk a few minutes after they are born, so they can follow the herd. Cattle are native to many parts of the world except the Americas, Australia and New Zealand. Cattle have been domesticated for about 9,000 years. They are used for milk, meat, transport, entertainment, and power.')
    elif prediction == 'sheep':
        st.write('A domestic sheep (Ovis aries) is a domesticated mammal related to wild sheep and goats. Sheep are owned and looked after by a sheep farmer. Female sheep are called ewes. Male sheep are called rams. Young sheep are called lambs. They are kept for their wool and their meat. The wool of sheep, after cleaning and treating, is used to make woollen clothes. The meat of young sheep is called lamb, and the meat from adult sheep is called mutton. Both are economically important products which have been used since prehistoric times. Sheep are domesticated animals which have been bred by man. There are breeds which specialise in wool or meat. The plural of sheep is just sheep.')
    elif prediction == 'spider':
        st.write('Spiders (class Arachnida, order Araneae) are air-breathing arthropods. They have eight legs, and mouthparts (chelicerae) with fangs that inject venom. Most make silk. The arachnids are seventh in number of species of all animal orders. About 48,000 spider species, and 120 families have been recorded by taxonomists. Over twenty different classifications have been proposed since 1900. Spiders live on every continent except for Antarctica, and in nearly every habitat with the exceptions of air and sea. Almost all spiders are predators, and most eat insects. They catch their prey in several ways. Some build a spider web, and some use a thread of silk that they throw at the insect. Some kinds of spiders hide in holes in the ground, then run out and grab an insect that walks by. Others will make web nets to throw at passing insects. Or they go out and simply attack their prey. Some can jump quite well and hunt by sneaking close to an insect and then jumping on it.')
    else:
        st.write('Squirrels are a large family of small to medium rodents. It includes tree squirrels, which are described on this page. The other squirrels are: ground squirrels, chipmunks, marmots (including groundhogs), flying squirrels, and prairie dogs. Squirrels are native to the Americas, Eurasia, and Africa, and have been introduced to Australia. The earliest known squirrels date from the Eocene and are most closely related to the mountain beaver and to the dormouse among living rodent families. Most squirrels are omnivores; they eat anything they find. Many kinds of squirrels live in trees, so they often find nuts. They eat seeds, berries and pine cones too. Sometimes they eat birds eggs and insects. Most tree squirrels store food in the fall, to eat in the winter. Ground squirrels do not store food. They hibernate which means they spend winter in a deep sleep. Squirrels have many predators or enemies. Foxes, wolves, coyotes, bears, raccoons, lynx, cougars, eagles, hawks and owls eat squirrels.')

if upload is not None:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
    img = Image.open(upload)
    st.image(img,caption='Uploaded Image',width=300)
    if st.sidebar.button('PREDICT'):
        st.sidebar.write("Result:")
        x = cv2.resize(opencv_image,(224,224))
        x = np.expand_dims(x,axis=0)
        # x = decode_predictions(x)
        result = model.predict(x)
        prediction = translate[np.argmax(result[0])]
        pred = classes[np.argmax(result[0])]
        st.write(f'predicted animal is {prediction}')
        st.write('Wiki:')
        info(prediction)
        image=Image.open('pos.png')
        st.image(image)
        


st.markdown("<h6 style='text-align: center; color:#7a000d;'>This is an Animla identification that identify 10 animal which are 'dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider'', 'squirrel'</h6>", unsafe_allow_html=True)


#    label = decode_predictions(y)
#    print the classification
#    for i in range(3):
#      out = label[0][i]
#      st.sidebar.title('%s (%.2f%%)' % (out[1], out[2]*100))