import pyttsx3

def speak(name)
	engine = pyttsx3.init()
	engine.say("hello {0}".format(name))
	engine.setProperty('rate',120)
	engine.setProperty('volume', 0.9)
	engine.runAndWait()