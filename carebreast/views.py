import os
import pyrebase
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm
from prediction.integrate import process_image  # Grad-CAM
from prediction.test import predict_image       # Prediction

# Firebase configuration
config = {
    'apiKey': "AIzaSyAM7b26IhvB75SvpN-bHaKHMtAArWJimuw",
    'authDomain': "breastcancerinsight.firebaseapp.com",
    'databaseURL': "https://breastcancerinsight-default-rtdb.firebaseio.com/",
    'projectId': "breastcancerinsight",
    'storageBucket': "breastcancerinsight.appspot.com",
    'messagingSenderId': "338793847759",
    'appId': "1:338793847759:web:d870384012e935507790ba",
    'measurementId': "G-QPGDD2LQTY"
}

firebase = pyrebase.initialize_app(config)
authe = firebase.auth()
database = firebase.database()

# Home / Welcome View
def welcome(request):
    is_authenticated = 'uid' in request.session
    return render(request, "welcome.html", {"is_authenticated": is_authenticated})

# FAQ View
def faq(request):
    return render(request, "faq.html")

# Image Upload + Prediction
def upload_image(request):
    form = ImageUploadForm()

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES['image']
            fs = FileSystemStorage()
            image_path = fs.save(uploaded_image.name, uploaded_image)
            full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)

            # 🔥 Run Prediction and Grad-CAM
            heatmap_path = process_image(full_image_path)
            label, probability = predict_image(full_image_path)

            # ✅ Format label nicely
            label_text = "Malignant" if label == 1 else "Benign"
            confidence = round(probability * 100, 2)

            # Set up URLs for displaying results
            heatmap_url = settings.MEDIA_URL + os.path.basename(heatmap_path)
            image_url = fs.url(image_path)

            # Render results
            context = {
                'image_url': image_url,
                'heatmap_url': heatmap_url,
                'label': label_text,
                'probability': confidence,
            }
            return render(request, 'result.html', context)

    return render(request, 'welcome.html', {'form': form})

# Sign-in Page
def signin(request):
    return render(request, "signin.html")

# Sign-in Handler
def postsign(request):
    email = request.POST.get('email')
    password = request.POST.get('password')
    try:
        user = authe.sign_in_with_email_and_password(email, password)
        uid = user['localId']
        request.session['uid'] = uid
        return redirect('welcome')
    except:
        message = "Invalid Credentials!"
        return render(request, "signin.html", {"mess": message})

# Logout
def logout(request):
    request.session.flush()
    return redirect('welcome')

# Sign-up Page
def signup(request):
    return render(request, "signup.html")

# Sign-up Handler
def postsignup(request):
    name = request.POST.get('name')
    email = request.POST.get('email')
    password = request.POST.get('password')

    try:
        user = authe.create_user_with_email_and_password(email, password)
        uid = user['localId']
        data = {
            "name": name,
            "email": email,
            "password": password,
            "status": "1"
        }
        database.child("users").child(uid).child("details").set(data)
        request.session['uid'] = uid
        return redirect('welcome')
    except:
        message = "Unable To Create Account as Password is too weak!"
        return render(request, "signup.html", {"mess": message})

# Account Details View
def account_details(request):
    if 'uid' not in request.session:
        return redirect('signin')
    
    uid = request.session['uid']
    try:
        user_details = database.child("users").child(uid).child("details").get().val()
    except Exception as e:
        print(f"Error fetching user details: {e}")
        user_details = None

    if not user_details:
        return render(request, "account_details.html", {"message": "No user details found"})

    return render(request, "account_details.html", {"user_details": user_details})
