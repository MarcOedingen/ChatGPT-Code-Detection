# Hacky solution to trick the pickle method to think there is a Utility module
# Need it to import the TFIDF vectorizer .pkl file
# Because we are using dill for pickling we the analyzer labmda function is also pickled