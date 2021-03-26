#include "Arduino.h"
#include "Ultrasound.h"

// Class Constructor
Ultrasound::Ultrasound(int pin)
{
  _pin = pin;
}

long Ultrasound::getDistance()
{
  // establish variables for duration of the ping, and the distance result in centimeters:
  long duration, cm; // long inches;

  pinMode(_pin, OUTPUT);
  // The PING is triggered by a HIGH pulse of 2 or more microseconds.
  // Give a short LOW pulse beforehand to ensure a clean HIGH pulse...
  
  // Short LOW pulse
  digitalWrite(_pin, LOW);
  delayMicroseconds(2);
  // Clean HIGH pulse
  digitalWrite(_pin, HIGH);
  delayMicroseconds(10);
  digitalWrite(_pin, LOW);

  // The same pin is used to read the signal from the PING: a HIGH pulse
  // whose duration is the time (in microseconds) from the sending of the ping
  // to the reception of its echo off of an object.
  pinMode(_pin, INPUT);
  duration = pulseIn(_pin, HIGH);

  // convert the time into a distance
  //inches = microsecondsToInches(duration);
  cm = microsecondsToCentimeters(duration);

  //Serial.print(inches); Serial.print("in, ");
  //Serial.print(cm); Serial.print("cm");
  //Serial.println();
  //delay(100);

  return cm;
}

long Ultrasound::microsecondsToInches(long microseconds) {
  // According to Parallax's datasheet for the PING))), there are 73.746
  // microseconds per inch (i.e. sound travels at 1130 feet per second).
  // This gives the distance travelled by the ping, outbound and return,
  // so we divide by 2 to get the distance of the obstacle.
  // See: http://www.parallax.com/dl/docs/prod/acc/28015-PING-v1.3.pdf
  return microseconds / 74 / 2;
}

long Ultrasound::microsecondsToCentimeters(long microseconds) {
  // The speed of sound is 340 m/s or 29 microseconds per centimeter.
  // The ping travels out and back, so to find the distance of the object we
  // take half of the distance travelled.
  return microseconds / 29 / 2;
}
