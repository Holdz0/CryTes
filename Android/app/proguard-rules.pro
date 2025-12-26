# Add project specific ProGuard rules here.
# TensorFlow Lite
-keep class org.tensorflow.** { *; }
-dontwarn org.tensorflow.**

# USB Serial
-keep class com.hoho.android.usbserial.** { *; }
-dontwarn com.hoho.android.usbserial.**
