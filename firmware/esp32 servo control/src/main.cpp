#include <Arduino.h>
#include <ESP32Servo.h>  



// Define Servo Pins
const int SERVO_PAN_PIN = 18; 
const int SERVO_TILT_PIN = 19; 

// Create Servo Objects
Servo panServo;
Servo tiltServo;
int currentPanAngle = 90;
int currentTiltAngle = 90;

void setup() {
  Serial.begin(115200); 

  // Allow allocation of all timers for ESP32 PWM
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  // Set standard servo properties of 50 Hz
  panServo.setPeriodHertz(50); 
  tiltServo.setPeriodHertz(50);

  // Attach servos to pins and specify min/max pulse widths in microseconds
  panServo.attach(SERVO_PAN_PIN, 500, 2400);
  tiltServo.attach(SERVO_TILT_PIN, 500, 2400);

  // Start at center position
  panServo.write(90);
  tiltServo.write(90);
}

void loop() {
  if (Serial.available() > 0) {
        // Read the incoming message until it hits a newline character

        String incomingData = Serial.readStringUntil('\n');
        
        // Trim whitespace or carriage returns just in case
        incomingData.trim();

        // Find the index of the separating comma
        int commaIndex = incomingData.indexOf(',');

        // Basic verification: Ensure a comma exists in the received string
        if (commaIndex != -1) {
            // Extract the pan and tilt substrings
            String panPart = incomingData.substring(0, commaIndex);
            String tiltPart = incomingData.substring(commaIndex + 1);

            // Convert string strings to integers
            int targetPan = panPart.toInt();
            int targetTilt = tiltPart.toInt();

          
            // This prevents the camera mount from binding or grinding gears
            targetPan = constrain(targetPan, 0, 175);
            targetTilt = constrain(targetTilt, 0, 175);

            // Update variables 
            currentPanAngle = targetPan;
            currentTiltAngle = targetTilt;
        }
  }

  // Write angles to the servos
  panServo.write(currentPanAngle);
  tiltServo.write(currentTiltAngle);

 

  delay(15);
}