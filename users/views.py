from django.shortcuts import render, redirect
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from .render import *
import shutil
from django.conf import settings

@login_required(login_url="http://127.0.0.1:8000/signin")
def home(request):
    # Delete files inside the uploads folder
    media_root = settings.MEDIA_ROOT
    try:
        shutil.rmtree(media_root)
    except Exception as e:
        print(f"Error deleting files and folders: {e}")

    # Recreate the empty directory
    os.makedirs(media_root, exist_ok=True)

    if request.method == 'POST' and request.FILES['image']:
        select_mra_or_cta = request.POST.get('selectMRAorCTA')
        select_label = request.POST.get('selectLabel')
        uploaded_file = request.FILES['image']

        original_image_path, predicted_file = predictor(uploaded_file, select_mra_or_cta, select_label)
        
        return render(request,"home.html", {'original_image_path': original_image_path, 'predicted_file': predicted_file})
    else:
        return render(request,"home.html")

def signin(request):
    return render(request,"signin.html")

def logout_view(request):
    logout(request)
    return redirect("http://127.0.0.1:8000/signin")