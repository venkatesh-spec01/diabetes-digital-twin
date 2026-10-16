# reference tables used by the doctor dashboard.
# this is general guidance for the treating doctor, based on common
# diabetes advice (for example the ADA standards of care).
# it is NOT a prescription and it does not calculate any dose.
# the hospital's own protocol and the doctor's judgement come first.
# the lists can be edited to match a hospital's protocol.
# glucose zones with colours for the bar
ZONES = [
    (0, 54, "Severe low", "#1d4ed8"),
    (54, 70, "Low", "#3b82f6"),
    (70, 180, "In target range", "#2ca02c"),
    (180, 250, "High", "#f59e0b"),
    (250, 400, "Very high", "#e4572e"),
]

ACTION_COLUMNS = ["Type", "Action", "When to do it", "How long", "Notes", "Who"]

# what to do when glucose is going high
ACTIONS_HIGH = [
    ("Lifestyle", "Easy walk at a normal pace", "After the meal (start 15-30 min after eating)", "10-15 minutes",
     "The what-if engine tests a 10-minute walk. Skip it if the patient is unwell or dizzy.", "Patient"),
    ("Lifestyle", "Drink water, avoid sugary drinks", "Right now", "Next 1-2 hours",
     "Dehydration can push glucose higher.", "Patient"),
    ("Diet", "About 30% smaller portion of rice, roti or other carbohydrate", "With the meal", "This meal",
     "Same change the what-if engine tests.", "Patient / dietitian"),
    ("Diet", "Swap to a lower-GI food: millet, ragi, whole-wheat roti or brown rice instead of white rice or biryani",
     "At the next meal", "Ongoing", "The what-if engine tests a swap to GI 50.", "Patient / dietitian"),
    ("Diet", "Start the meal with vegetables, dal or curd before the carbohydrate", "With the meal", "This meal",
     "Often slows the rise after the meal.", "Patient"),
    ("Monitoring", "Recheck glucose", "1-2 hours later, and at bedtime if it was high in the evening", "Until back in range",
     "", "Patient"),
    ("Monitoring", "Check ketones if glucose stays above 250 mg/dL with nausea, vomiting, stomach pain or fast breathing",
     "Right now", "Same day", "Needs urgent medical care if ketones are present.", "Patient / doctor"),
    ("Medicine (doctor decides)", "Review the timing of the current medicines: metformin with or just after meals, "
     "sulfonylurea with the first main meal, mealtime insulin as prescribed before the meal",
     "Before the meal", "Per prescription", "No dose is suggested here. Changes only by the treating doctor.", "Doctor"),
    ("Medicine (doctor decides)", "Correction insulin only if one is already prescribed, following the patient's written plan",
     "As written in the plan", "Per prescription", "No dose is suggested here.", "Doctor"),
    ("Follow-up", "If forecasts stay high for several days, review HbA1c, missed doses and the regimen",
     "At the next visit", "Next visit", "", "Doctor"),
]

# what to do when glucose is going low
ACTIONS_LOW = [
    ("Immediate", "15 g of fast sugar: 3-4 glucose tablets, or about 150 mL fruit juice, or 3-4 teaspoons of sugar in water",
     "Right now", "0-15 minutes", "Avoid chocolate and fatty foods, they act slowly.", "Patient"),
    ("Immediate", "Recheck after 15 minutes. If still below 70 mg/dL, repeat the fast sugar", "Right now",
     "Every 15 minutes until above 70", "", "Patient"),
    ("Follow-up", "Once above 70, eat a small snack with slow carbohydrate and protein if the next meal is more than an hour away "
     "(for example roti with dal, roasted chana, curd or a fruit)", "After correcting", "15-60 minutes", "", "Patient"),
    ("Meal timing", "Do not skip or delay the meal. Take sulfonylurea or insulin only with food", "Before the meal", "Ongoing",
     "Skipped meals are a common cause of lows.", "Patient"),
    ("Activity", "Pause planned exercise until glucose recovers, and carry fast sugar when active", "Any time", "Ongoing", "",
     "Patient"),
    ("Monitoring", "Check glucose before driving, before sleeping and before exercise", "Any time", "Ongoing", "", "Patient"),
    ("Severe low", "If the patient is confused, cannot swallow, has a fit or is unconscious: nothing by mouth. "
     "Give glucagon if prescribed and call emergency services", "Right now", "Emergency", "", "Carer / doctor"),
    ("Medicine (doctor decides)", "If lows repeat, review the sulfonylurea or insulin dose and timing", "At the next visit",
     "Next visit", "No dose is suggested here.", "Doctor"),
]

# groups used to filter by timing
TIMING_FILTER = ["Right now", "Before the meal", "With the meal", "After the meal", "Any time", "Later / next visit"]


# puts a timing text into a group
def timing_group(text):
    t = text.lower()
    if t.startswith("right now"):
        return "Right now"
    if t.startswith("before"):
        return "Before the meal"
    if t.startswith("with the meal") or t.startswith("at the next meal"):
        return "With the meal"
    if t.startswith("after the meal") or t.startswith("after correcting"):
        return "After the meal"
    if t.startswith("any time"):
        return "Any time"
    return "Later / next visit"


# sample indian diet plans by calories
DIET_PLANS = {
    "1400 kcal": [
        ("Early morning", "6:30", "Methi (fenugreek) water or plain water, 4-5 soaked almonds", "1 glass + 5 almonds", 2, "Low"),
        ("Breakfast", "8:00", "Ragi dosa with sambar and chutney", "1 dosa + 1 katori sambar", 30, "Low-medium"),
        ("Mid-morning", "11:00", "Guava, papaya or apple", "1 small fruit", 10, "Low"),
        ("Lunch", "13:00", "Millet or whole-wheat roti, dal, vegetable sabzi, salad, curd", "1 roti + 1 katori dal + sabzi + curd", 45, "Low-medium"),
        ("Evening", "17:00", "Roasted chana with unsweetened tea", "1 handful chana", 12, "Low"),
        ("Dinner", "20:00", "Millet khichdi or roti with dal and sabzi", "1 small katori khichdi or 1 roti", 40, "Low-medium"),
        ("Bedtime (if hungry)", "22:00", "Warm milk, no sugar", "1 small glass", 8, "Low"),
    ],
    "1600 kcal": [
        ("Early morning", "6:30", "Methi (fenugreek) water or plain water, 4-5 soaked almonds", "1 glass + 5 almonds", 2, "Low"),
        ("Breakfast", "8:00", "Ragi dosa with sambar and chutney", "2 dosa + 1 katori sambar", 40, "Low-medium"),
        ("Mid-morning", "11:00", "Guava, papaya or apple", "1 medium fruit", 15, "Low"),
        ("Lunch", "13:00", "Millet or whole-wheat roti, dal, vegetable sabzi, salad, curd", "2 roti + 1 katori dal + sabzi + curd", 55, "Low-medium"),
        ("Evening", "17:00", "Roasted chana with unsweetened tea", "1 handful chana", 15, "Low"),
        ("Dinner", "20:00", "Millet khichdi or roti with dal and sabzi", "1 katori khichdi or 2 roti", 50, "Low-medium"),
        ("Bedtime (if hungry)", "22:00", "Warm milk, no sugar", "1 small glass", 8, "Low"),
    ],
    "1800 kcal": [
        ("Early morning", "6:30", "Methi (fenugreek) water or plain water, 4-5 soaked almonds", "1 glass + 5 almonds", 2, "Low"),
        ("Breakfast", "8:00", "Ragi dosa or oats upma with sambar and chutney", "2 dosa + 1 katori sambar", 45, "Low-medium"),
        ("Mid-morning", "11:00", "Guava, papaya or apple", "1 medium fruit + 1 handful nuts", 15, "Low"),
        ("Lunch", "13:00", "Millet or whole-wheat roti, dal, vegetable sabzi, salad, curd", "3 roti + 1 katori dal + sabzi + curd", 65, "Low-medium"),
        ("Evening", "17:00", "Roasted chana with unsweetened tea", "1 handful chana + 1 fruit", 20, "Low"),
        ("Dinner", "20:00", "Millet khichdi or roti with dal and sabzi", "1 large katori khichdi or 2 roti", 60, "Low-medium"),
        ("Bedtime (if hungry)", "22:00", "Warm milk, no sugar", "1 small glass", 8, "Low"),
    ],
}
# column names for the diet table
DIET_COLUMNS = ["Meal", "Time", "Food", "Portion", "Carbs (g, approx.)", "GI band"]

# simple rules given with the diet chart
DIET_RULES = [
    "Eat at fixed times and do not skip meals, especially on insulin or a sulfonylurea (risk of lows).",
    "Fill half the plate with vegetables, a quarter with dal, egg, fish, paneer or chicken, and a quarter with grain.",
    "Prefer millets (ragi, jowar, bajra), whole wheat and brown rice over white rice, maida and biryani.",
    "Avoid sweets, sugary drinks, fruit juice and fried snacks. Choose whole fruit instead.",
    "A 10-15 minute easy walk after meals helps.",
]


# picks a diet plan from the bmi
def default_plan_for(bmi):
    if bmi >= 30:
        return "1400 kcal"
    if bmi < 23:
        return "1800 kcal"
    return "1600 kcal"
