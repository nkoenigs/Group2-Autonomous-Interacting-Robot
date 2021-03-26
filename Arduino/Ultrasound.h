#ifndef Ultrasound_h
#define Ultrasound_h

#include "Arduino.h"

// Header file for Ultrasound.cpp

class Ultrasound
{
  public:
    Ultrasound(int pin);
    long getDistance();
    long microsecondsToInches(long microseconds);
    long microsecondsToCentimeters(long microseconds);
  private:
    int _pin;
};

#endif
