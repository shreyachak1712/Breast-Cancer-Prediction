"""
URL configuration for carebreast project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from tkinter.font import names

from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

from .views import welcome, upload_image


urlpatterns = [
    path('admin/', admin.site.urls),  # Admin route
   path('upload_image/', upload_image, name='upload_image'),
    path('', views.welcome, name='welcome'),  # Welcome page (default)
    path('signin/', views.signin, name='signin'),  # Sign in page
    path('postsign/', views.postsign, name='postsign'),  # Post sign in action
    path('signup/', views.signup, name='signup'),  # Sign up page
    path('postsignup/', views.postsignup, name='postsignup'),  # Post sign up action
    path('logout/', views.logout, name='logout'),  # Logout action
    path('faq/', views.faq, name='faq'),  # FAQ in page
    path('account/', views.account_details, name='account_details'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# Serving the media files in development mode
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
else:
    urlpatterns += staticfiles_urlpatterns()