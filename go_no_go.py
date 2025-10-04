import pandas as pd
import joblib

# Load saved models
conv_model = joblib.load("conv_model.pkl")
fg_model = joblib.load("fg_model.pkl")

def fourth_down_decider(ydstogo, yardline_100, kick_distance):
    # Go for it probability
    go_prob = conv_model.predict_proba(
        pd.DataFrame([[ydstogo, yardline_100]], columns=['ydstogo','yardline_100'])
    )[0,1]

    # Field goal probability
    fg_prob = fg_model.predict_proba(
        pd.DataFrame([[kick_distance]], columns=['kick_distance'])
    )[0,1]

    # -----------------------
    # Crude Expected Points placeholders
    # -----------------------
    ep_success = 4.5
    ep_fail = -1.0
    go_ep = go_prob * ep_success + (1-go_prob) * ep_fail

    punt_ep = -0.7
    fg_ep = fg_prob * 3 + (1-fg_prob) * (-0.7)

    # Choose best option
    options = {"Go for it": go_ep, "Punt": punt_ep, "Field Goal": fg_ep}
    best_choice = max(options, key=options.get)

    return {
        "go_prob": go_prob,
        "fg_prob": fg_prob,
        "EP_go": go_ep,
        "EP_punt": punt_ep,
        "EP_fg": fg_ep,
        "best_choice": best_choice
    }


ydstogo = 3
yardline_100 = 25
kick_distance = yardline_100 + 17
print(f"4th and {ydstogo} from the {yardline_100}")
print((fourth_down_decider(ydstogo, yardline_100, kick_distance))['best_choice'])

ydstogo = 2
yardline_100 = 25
kick_distance = yardline_100 + 17
print(f"4th and {ydstogo} from the {yardline_100}")
print((fourth_down_decider(ydstogo, yardline_100, kick_distance))['best_choice'])

ydstogo = 1
yardline_100 = 25
kick_distance = yardline_100 + 17
print(f"4th and {ydstogo} from the {yardline_100}")
print((fourth_down_decider(ydstogo, yardline_100, kick_distance))['best_choice'])



ydstogo = 1
yardline_100 = 1
kick_distance = yardline_100 + 17
print(f"4th and {ydstogo} from the {yardline_100}")
print((fourth_down_decider(ydstogo, yardline_100, kick_distance))['best_choice'])

ydstogo = 2
yardline_100 = 2
kick_distance = yardline_100 + 17
print(f"4th and {ydstogo} from the {yardline_100}")
print((fourth_down_decider(ydstogo, yardline_100, kick_distance))['best_choice'])

ydstogo = 3
yardline_100 = 3
kick_distance = yardline_100 + 17
print(f"4th and {ydstogo} from the {yardline_100}")
print((fourth_down_decider(ydstogo, yardline_100, kick_distance))['best_choice'])
