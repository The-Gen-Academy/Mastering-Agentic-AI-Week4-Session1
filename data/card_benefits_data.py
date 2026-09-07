"""
CardWise AI - Credit Card Benefits Data Module
Contains structured inline Python list of benefit documents for:
1. American Express Platinum Card
2. Chase Sapphire Reserve
"""

CARD_BENEFITS = [
    # ==========================================
    # AMERICAN EXPRESS PLATINUM CARD BENEFITS
    # ==========================================
    {
        "id": "amex_plat_overview",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Card Overview & Annual Fee",
        "category": "Card Overview",
        "annual_credit_value": 0,
        "content": (
            "The Platinum Card® from American Express is a premium travel and lifestyle rewards card issued by American Express. "
            "The card provides Membership Rewards® points, airport lounge access, hotel benefits, travel credits, dining benefits, "
            "entertainment credits, shopping and wellness benefits, travel insurance, and purchase protections. "
            "The annual fee for the Platinum Card is $895. Benefit eligibility and enrollment requirements vary. "
            "Some benefits require cardmembers to enroll before receiving the benefit."
        ),
        "keywords": ["annual fee", "$895", "overview", "enrollment", "american express", "amex platinum"]
    },
    {
        "id": "amex_plat_welcome_offer",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Welcome Offer",
        "category": "Welcome Offer",
        "annual_credit_value": 0,
        "content": (
            "Eligible new cardmembers may receive a welcome offer of Membership Rewards points after meeting a specified spending requirement. "
            "The publicly advertised offer can be as high as 175,000 Membership Rewards points after spending $12,000 in eligible purchases "
            "within the first six months of Card Membership. The exact welcome offer presented to an applicant may vary, and not every "
            "applicant will receive the maximum advertised offer. Eligibility is subject to American Express rules."
        ),
        "keywords": ["welcome offer", "sign-up bonus", "175,000 points", "$12,000 spend", "6 months"]
    },
    {
        "id": "amex_plat_rewards_earning",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Rewards Earning Multipliers",
        "category": "Rewards Earning",
        "annual_credit_value": 0,
        "multipliers": {"flights": 5, "prepaid_hotels": 5, "other": 1},
        "content": (
            "Cardmembers earn Membership Rewards points on eligible purchases: "
            "1) Flights: Earn 5 Membership Rewards points per dollar on eligible flights booked directly with airlines or through American Express Travel, "
            "up to $500,000 in eligible flight purchases per calendar year (1X thereafter). "
            "2) Prepaid Hotels: Earn 5 Membership Rewards points per dollar on eligible prepaid hotel reservations booked through American Express Travel. "
            "3) Other Purchases: Other eligible purchases generally earn 1 Membership Rewards point per dollar."
        ),
        "keywords": ["rewards earning", "multipliers", "5x flights", "5x hotels", "1x points", "$500k cap"]
    },
    {
        "id": "amex_plat_hotel_credit",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "$600 Hotel Credit",
        "category": "Travel Credits",
        "annual_credit_value": 600,
        "content": (
            "Cardmembers can receive up to $600 per calendar year in statement credits for qualifying prepaid hotel bookings through "
            "American Express Travel. The benefit is divided into two semiannual periods: up to $300 in statement credits during each eligible "
            "six-month period (Jan-Jun, Jul-Dec). Qualifying reservations must be prepaid bookings through Fine Hotels + Resorts® or "
            "The Hotel Collection (which requires a minimum stay of two nights)."
        ),
        "keywords": ["$600 hotel credit", "$300 semiannual", "fine hotels + resorts", "the hotel collection", "amex travel"]
    },
    {
        "id": "amex_plat_fhr",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Fine Hotels + Resorts Program",
        "category": "Hotel Benefits & Status",
        "annual_credit_value": 0,
        "content": (
            "Platinum Card members have access to more than 1,800 Fine Hotels + Resorts properties worldwide booked through Amex Travel. "
            "Eligible reservations receive premium benefits: daily breakfast for two, room upgrade upon arrival when available, "
            "a $100 property credit toward eligible charges (e.g. spa/dining), guaranteed 4:00 PM late checkout, and 12:00 PM early check-in when available. "
            "Average estimated benefit value is over $550 per stay for two-night stays."
        ),
        "keywords": ["fine hotels resorts", "fhr", "$100 property credit", "breakfast for two", "late checkout"]
    },
    {
        "id": "amex_plat_hotel_collection",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "The Hotel Collection",
        "category": "Hotel Benefits & Status",
        "annual_credit_value": 0,
        "content": (
            "Access to more than 1,300 participating upscale hotels worldwide. Qualifying reservations require a minimum stay of two nights "
            "booked through American Express Travel. Eligible bookings receive a $100 credit toward qualifying charges at the property and a room upgrade upon arrival when available."
        ),
        "keywords": ["hotel collection", "$100 credit", "2 night minimum", "room upgrade"]
    },
    {
        "id": "amex_plat_lounge_access",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Global Lounge Collection",
        "category": "Lounge Access",
        "annual_credit_value": 0,
        "content": (
            "Platinum Card members receive access to the American Express Global Lounge Collection with more than 1,550 airport lounges worldwide. "
            "Includes Centurion Lounges, Priority Pass lounges (enrollment required), Delta Sky Club access when traveling on an eligible Delta flight "
            "(limited to 10 Delta Sky Club visits per year), and other participating partner lounges."
        ),
        "keywords": ["global lounge collection", "centurion lounge", "priority pass", "delta sky club", "airport lounges"]
    },
    {
        "id": "amex_plat_airline_fee_credit",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "$200 Airline Fee Credit",
        "category": "Travel Credits",
        "annual_credit_value": 200,
        "content": (
            "Cardmembers can receive up to $200 per calendar year in statement credits for qualifying incidental airline fees on one selected airline. "
            "Eligible charges include checked baggage fees, in-flight refreshments, flight-change fees, overweight/oversized baggage fees, "
            "airport lounge day passes, pet kennel fees, and phone reservation fees. Does NOT cover base airline ticket price."
        ),
        "keywords": ["$200 airline fee credit", "incidental charges", "baggage fees", "lounge passes", "selected airline"]
    },
    {
        "id": "amex_plat_uber_cash",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "$200 Uber Cash",
        "category": "Travel & Transportation Credits",
        "annual_credit_value": 200,
        "content": (
            "Cardmembers receive up to $200 per year in Uber Cash. Distributed as $15 in Uber Cash each month from Jan through Nov, and $35 in Dec ($15 monthly + $20 Dec bonus). "
            "Valid for qualifying Uber rides and Uber Eats orders in the United States. Platinum Card must be added to Uber account and selected as payment method."
        ),
        "keywords": ["$200 uber cash", "$15 monthly", "$35 december", "uber eats", "uber rides"]
    },
    {
        "id": "amex_plat_uber_one_credit",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "$120 Uber One Credit",
        "category": "Travel & Transportation Credits",
        "annual_credit_value": 120,
        "content": (
            "Up to $120 per calendar year in statement credits toward an eligible auto-renewing Uber One membership purchased with the Platinum Card. "
            "This benefit is separate from the $200 annual Uber Cash benefit."
        ),
        "keywords": ["$120 uber one credit", "uber one membership", "statement credit"]
    },
    {
        "id": "amex_plat_clear_credit",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "CLEAR+ Credit",
        "category": "Travel Credits",
        "annual_credit_value": 219,
        "content": (
            "Up to $219 per calendar year in statement credits toward an eligible CLEAR+ membership purchased using the Platinum Card. "
            "Applies to auto-renewing CLEAR+ membership."
        ),
        "keywords": ["clear+ credit", "$219 clear", "airport security", "expedited security"]
    },
    {
        "id": "amex_plat_global_entry_tsa",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Global Entry or TSA PreCheck Credit",
        "category": "Travel Credits",
        "annual_credit_value": 120,
        "content": (
            "Fee credit toward an eligible Global Entry or TSA PreCheck application fee charged to the Platinum Card. "
            "Credit applies once every 4 to 4.5 years depending on program terms."
        ),
        "keywords": ["global entry", "tsa precheck", "application fee credit", "trusted traveler"]
    },
    {
        "id": "amex_plat_resy_credit",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "$400 Resy Credit",
        "category": "Dining Credits",
        "annual_credit_value": 400,
        "content": (
            "Up to $400 per calendar year in statement credits for qualifying Resy purchases. Divided quarterly: up to $100 in statement credits each quarter "
            "(Q1, Q2, Q3, Q4). Valid for dining purchases at over 10,000 participating U.S. Resy restaurants and eligible Resy purchases. Enrollment required."
        ),
        "keywords": ["$400 resy credit", "$100 quarterly", "resy dining", "restaurants", "resy enrollment"]
    },
    {
        "id": "amex_plat_resy_access",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Global Dining Access & Platinum Nights by Resy",
        "category": "Dining Benefits",
        "annual_credit_value": 0,
        "content": (
            "Special access to participating restaurants through Global Dining Access by Resy: exclusive reservations, Priority Notify, "
            "and premium dining experiences. Platinum Nights by Resy offers access to sought-after reservations on selected nights in major cities. "
            "Card must be linked to Resy profile."
        ),
        "keywords": ["global dining access", "resy", "platinum nights", "exclusive reservations", "priority notify"]
    },
    {
        "id": "amex_plat_digital_entertainment",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "$300 Digital Entertainment Credit",
        "category": "Lifestyle & Shopping Credits",
        "annual_credit_value": 300,
        "content": (
            "Up to $300 per calendar year in statement credits for qualifying digital entertainment purchases ($25/month). "
            "Eligible participating services: Disney+, Disney+ bundles, ESPN+, Hulu, The New York Times, Paramount+, Peacock, The Wall Street Journal, "
            "YouTube Premium, YouTube TV. Direct provider subscription required. Enrollment required."
        ),
        "keywords": ["$300 digital entertainment credit", "$25 monthly", "disney+", "hulu", "paramount+", "peacock", "wsj", "youtube tv", "nyt"]
    },
    {
        "id": "amex_plat_lululemon",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "$300 lululemon Credit",
        "category": "Lifestyle & Shopping Credits",
        "annual_credit_value": 300,
        "content": (
            "Up to $300 per calendar year in statement credits for qualifying lululemon purchases. Divided quarterly: up to $75 per quarter. "
            "Valid at U.S. lululemon retail stores (excluding outlets) and online at lululemon.com. Enrollment required."
        ),
        "keywords": ["$300 lululemon credit", "$75 quarterly", "lululemon", "shopping credit"]
    },
    {
        "id": "amex_plat_walmart_plus",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Walmart+ Membership Credit",
        "category": "Lifestyle & Shopping Credits",
        "annual_credit_value": 155,
        "content": (
            "Statement credit covering up to $12.95 per month plus applicable taxes for an eligible monthly Walmart+ membership paid with the Platinum Card "
            "(~ $155 annual value). Subject to auto-renewal."
        ),
        "keywords": ["walmart+ credit", "walmart plus", "$12.95 monthly", "grocery delivery"]
    },
    {
        "id": "amex_plat_oura_ring",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "$200 Oura Ring Credit",
        "category": "Lifestyle & Wellness Credits",
        "annual_credit_value": 200,
        "content": (
            "Up to $200 per calendar year in statement credits toward an eligible Oura Ring purchase made through ouraring.com using the Platinum Card. "
            "Enrollment required."
        ),
        "keywords": ["$200 oura ring credit", "oura ring", "health tracking", "wellness credit"]
    },
    {
        "id": "amex_plat_equinox",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "$300 Equinox Credit",
        "category": "Lifestyle & Wellness Credits",
        "annual_credit_value": 300,
        "content": (
            "Up to $300 per calendar year in statement credits for qualifying Equinox club memberships or Equinox+ digital subscriptions paid using the Platinum Card. "
            "Enrollment required."
        ),
        "keywords": ["$300 equinox credit", "equinox fitness", "equinox+", "gym membership credit"]
    },
    {
        "id": "amex_plat_hotel_statuses",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Hotel Elite Statuses (Marriott, Hilton, LHW)",
        "category": "Hotel Benefits & Status",
        "annual_credit_value": 0,
        "content": (
            "Complimentary hotel elite statuses upon enrollment: "
            "1) Marriott Bonvoy Gold Elite Status (room upgrades when available). "
            "2) Hilton Honors Gold Status (space-available upgrades, 80% point bonus, daily food & beverage credit / continental breakfast, 5th night free on reward stays). "
            "3) Leading Hotels of the World (Leaders Club Sterling Status with 5% point bonus and 5 pre-arrival upgrades)."
        ),
        "keywords": ["marriott gold elite", "hilton honors gold", "leading hotels of the world", "hotel status", "room upgrades"]
    },
    {
        "id": "amex_plat_car_rental",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Car Rental Status & Insurance",
        "category": "Car Rental & Travel Protection",
        "annual_credit_value": 0,
        "content": (
            "Car Rental Privileges: Elite status privileges with Avis, Hertz, and National. "
            "Car Rental Loss and Damage Insurance: Secondary coverage for theft or damage to an eligible rental vehicle when paid with Platinum Card and collision damage waiver is declined. "
            "Excludes liability. Certain countries (Australia, Italy, New Zealand) excluded."
        ),
        "keywords": ["car rental privileges", "avis", "hertz", "national", "secondary collision insurance"]
    },
    {
        "id": "amex_plat_travel_insurance",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Trip Delay & Cancellation/Interruption Insurance",
        "category": "Insurance & Protections",
        "annual_credit_value": 0,
        "content": (
            "1) Trip Delay Insurance: Reimburses up to $500 per eligible trip when delayed over 6 hours for covered reasons. Max 2 claims per 12-month period. "
            "2) Trip Cancellation and Interruption Insurance: Reimburses up to $10,000 per trip (max $20,000 per 12 months) for non-refundable expenses when cancelled/interrupted for covered reasons. "
            "Trip must be paid entirely with Platinum Card."
        ),
        "keywords": ["trip delay insurance", "$500 delay", "6 hour delay", "trip cancellation insurance", "$10,000 coverage"]
    },
    {
        "id": "amex_plat_purchase_protections",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Purchase Protection, Extended Warranty, Return Protection & Cell Phone Protection",
        "category": "Insurance & Protections",
        "annual_credit_value": 0,
        "content": (
            "1) Cell Phone Protection: Up to $800 per approved claim (max 2 claims/$50 deductible per 12 months) for damage/theft when wireless bill is paid with card. "
            "2) Purchase Protection: Up to $10,000 per purchase ($50,000 max/yr) against damage, theft or loss within 90 days. "
            "3) Return Protection: Up to $300 per item ($1,000 max/yr) within 90 days if merchant refuses return. "
            "4) Extended Warranty: Adds up to 1 extra year on eligible original warranties of 5 years or less (up to $10,000/item, $50k/yr). "
            "5) Foreign Transactions: $0 foreign transaction fees."
        ),
        "keywords": ["cell phone protection", "$800 claim", "purchase protection", "return protection", "extended warranty", "no foreign fees"]
    },
    {
        "id": "amex_plat_redemption",
        "card": "American Express Platinum Card",
        "card_code": "amex_plat",
        "topic": "Membership Rewards Point Redemption",
        "category": "Redemption & Transfer",
        "annual_credit_value": 0,
        "content": (
            "Membership Rewards points can be redeemed for flights booked through Amex Travel, prepaid hotels through Amex Travel, "
            "Pay with Points at participating partners, gift cards, and airline/hotel transfer partners. Redemption value varies based on method."
        ),
        "keywords": ["membership rewards redemption", "amex travel points", "point value", "transfer partners"]
    },

    # ==========================================
    # CHASE SAPPHIRE RESERVE BENEFITS
    # ==========================================
    {
        "id": "csr_overview",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "Card Overview & Annual Fee",
        "category": "Card Overview",
        "annual_credit_value": 0,
        "content": (
            "The Chase Sapphire Reserve is a premium travel rewards credit card issued by Chase. "
            "The card provides travel rewards, airport lounge access, travel and lifestyle statement credits, dining benefits, and premium services. "
            "The annual fee is $795. Adding an authorized user costs $195 per authorized user."
        ),
        "keywords": ["chase sapphire reserve", "annual fee", "$795", "authorized user $195", "overview"]
    },
    {
        "id": "csr_welcome_offer",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "New Cardmember Bonus",
        "category": "Welcome Offer",
        "annual_credit_value": 0,
        "content": (
            "Eligible new cardmembers can earn 100,000 Ultimate Rewards points after spending $6,000 on qualifying purchases "
            "during the first three months after opening the account. 100,000 points can be worth up to $2,000 when redeemed for eligible flights and hotels "
            "through Chase Travel using qualifying Points Boost offers. Subject to Chase 5/24 and bonus eligibility rules."
        ),
        "keywords": ["welcome bonus", "100,000 points", "$6,000 spend", "3 months", "$2,000 value"]
    },
    {
        "id": "csr_rewards_earning",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "Rewards Earning Multipliers",
        "category": "Rewards Earning",
        "annual_credit_value": 0,
        "multipliers": {"chase_travel": 8, "direct_flights": 4, "direct_hotels": 4, "dining": 3, "other": 1},
        "content": (
            "Cardmembers earn Ultimate Rewards points on purchases: "
            "1) Chase Travel: 8 points per dollar on eligible flights, hotels, car rentals, cruises, and tours booked through Chase Travel. "
            "2) Direct Flights: 4 points per dollar on flights booked directly with airlines. "
            "3) Direct Hotels: 4 points per dollar on hotels booked directly with hotels. "
            "4) Dining: 3 points per dollar on restaurants, takeout, and qualifying delivery services. "
            "5) Other Purchases: 1 point per dollar. "
            "Note: Purchases reimbursed by statement credits (like the $300 Travel Credit) do not earn points."
        ),
        "keywords": ["csr rewards earning", "8x chase travel", "4x flights", "4x hotels", "3x dining", "1x points"]
    },
    {
        "id": "csr_annual_travel_credit",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "$300 Annual Travel Credit",
        "category": "Travel Credits",
        "annual_credit_value": 300,
        "content": (
            "Cardmembers receive up to $300 in statement credits for eligible travel purchases each account anniversary year. "
            "The credit automatically applies to qualifying travel purchases including airlines, hotels, car rentals, cruise lines, "
            "passenger trains, buses, taxis, ferries, tolls, and parking. Based on account year rather than calendar year."
        ),
        "keywords": ["$300 annual travel credit", "automatic travel credit", "parking", "tolls", "trains", "account anniversary"]
    },
    {
        "id": "csr_chase_travel_hotel_credit_2026",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "Chase Travel Hotel Credit (2026)",
        "category": "Travel Credits",
        "annual_credit_value": 250,
        "content": (
            "Through December 31, 2026, eligible cardmembers can receive up to $250 in statement credits for qualifying prepaid Chase Travel hotel reservations. "
            "Eligible brands include IHG Hotels & Resorts, Montage, Pendry, Omni, Virgin Hotels, Minor Hotels, Pan Pacific. "
            "Requires a prepaid 'Pay Now' stay of at least two nights. Limited-time 2026 benefit."
        ),
        "keywords": ["$250 hotel credit", "chase travel 2026 credit", "ihg", "montage", "2 night minimum"]
    },
    {
        "id": "csr_the_edit_hotel_credit",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "The Edit Hotel Credit & Benefits",
        "category": "Hotel Benefits & Status",
        "annual_credit_value": 500,
        "content": (
            "1) Up to $500 per calendar year in statement credits for qualifying prepaid hotel stays booked through The Edit by Chase Travel "
            "(structured as up to $250 per booking, max $500/yr, 2-night minimum stay required). "
            "2) Hotel Benefits: Daily breakfast for two, property credit ($100), room upgrades when available, and special amenities "
            "(average estimated value over $550 per eligible stay)."
        ),
        "keywords": ["the edit hotel credit", "$500 edit credit", "$250 per stay", "breakfast for two", "$100 property credit"]
    },
    {
        "id": "csr_lounge_access",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "Airport Lounge Access (Sapphire Lounge & Priority Pass)",
        "category": "Lounge Access",
        "annual_credit_value": 0,
        "content": (
            "Complimentary access to participating Chase Sapphire Lounges by The Club. "
            "Also includes complimentary Priority Pass Select membership giving access to over 1,300 airport lounges worldwide. "
            "Cardmembers can bring up to two guests free of charge per visit."
        ),
        "keywords": ["chase sapphire lounge", "priority pass select", "lounge access", "2 free guests"]
    },
    {
        "id": "csr_global_entry_tsa_nexus",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "Global Entry, TSA PreCheck or NEXUS Credit",
        "category": "Travel Credits",
        "annual_credit_value": 120,
        "content": (
            "Statement credit of up to $120 every 4 years toward an eligible application fee for Global Entry, TSA PreCheck, or NEXUS "
            "charged to the Sapphire Reserve card."
        ),
        "keywords": ["global entry", "tsa precheck", "nexus credit", "$120 credit"]
    },
    {
        "id": "csr_dining_credit_opentable",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "$300 OpenTable Sapphire Exclusive Dining Credit",
        "category": "Dining Credits",
        "annual_credit_value": 300,
        "content": (
            "Up to $300 per calendar year in statement credits for qualifying dining purchases at restaurants participating in the "
            "Sapphire Exclusive Tables program on OpenTable. Divided into two 6-month periods: $150 from Jan-Jun and $150 from Jul-Dec."
        ),
        "keywords": ["$300 opentable credit", "sapphire exclusive tables", "$150 semiannual", "dining credit"]
    },
    {
        "id": "csr_doordash",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "DoorDash DashPass & Monthly Promo Value",
        "category": "Dining & Delivery Credits",
        "annual_credit_value": 300,
        "content": (
            "Complimentary DashPass membership ($0 delivery fee and reduced service fees). "
            "Includes up to $25 per month in DoorDash promo value: a $5 monthly promo for restaurant orders, and two $10 monthly promos for "
            "grocery, retail, and non-restaurant orders. Benefit scheduled through Dec 31, 2027."
        ),
        "keywords": ["doordash", "dashpass", "$25 monthly promo", "$5 restaurant", "two $10 grocery promos"]
    },
    {
        "id": "csr_apple_entertainment",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "Apple TV+ and Apple Music Subscriptions",
        "category": "Lifestyle & Entertainment",
        "annual_credit_value": 288,
        "content": (
            "Complimentary subscriptions to Apple TV+ and Apple Music (~ $288 annual value). "
            "Subject to activation and eligibility rules. Benefit scheduled through 2027."
        ),
        "keywords": ["apple tv+", "apple music", "free apple subscriptions", "$288 value"]
    },
    {
        "id": "csr_stubhub_viagogo",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "$300 StubHub and viagogo Credit",
        "category": "Lifestyle & Entertainment",
        "annual_credit_value": 300,
        "content": (
            "Up to $300 per calendar year in statement credits for eligible purchases on StubHub and viagogo. "
            "Divided semiannually: $150 from Jan-Jun and $150 from Jul-Dec. Activation required. Scheduled through Dec 31, 2027."
        ),
        "keywords": ["$300 stubhub credit", "viagogo", "concert tickets", "event tickets", "$150 semiannual"]
    },
    {
        "id": "csr_lyft_benefits",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "Lyft Monthly Credits & 5X Multipliers",
        "category": "Travel & Transportation Credits",
        "annual_credit_value": 120,
        "content": (
            "Up to $10 per month in Lyft in-app credits ($120/year max). "
            "Eligible Lyft rides also earn 5 Ultimate Rewards points per dollar. Scheduled through Sept 30, 2027."
        ),
        "keywords": ["lyft credit", "$10 monthly lyft", "5x points lyft", "rideshare"]
    },
    {
        "id": "csr_peloton_benefits",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "Peloton Statement Credits & 10X Points",
        "category": "Lifestyle & Wellness Credits",
        "annual_credit_value": 120,
        "content": (
            "Up to $10 per month in statement credits toward eligible Peloton memberships ($120/yr max). "
            "Also earn 10 points per dollar on eligible Peloton equipment/accessories over $150. Activation required. Scheduled through Dec 31, 2027."
        ),
        "keywords": ["peloton credit", "$10 monthly peloton", "10x peloton equipment", "fitness"]
    },
    {
        "id": "csr_points_boost_redemption",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "Ultimate Rewards, Points Boost & Transfer Partners",
        "category": "Redemption & Transfer",
        "annual_credit_value": 0,
        "content": (
            "1) Points Boost: Points redeemed through Chase Travel can get elevated value up to 2.0 cents per point for select flights and hotels "
            "(standard baseline 1.0 cent/point). "
            "2) Point Transfers: 1:1 transfer to 14+ leading airline and hotel partner programs (e.g. Hyatt, United, Southwest, British Airways). "
            "3) Points do not expire as long as account remains open."
        ),
        "keywords": ["points boost", "2.0 cents per point", "transfer partners", "1:1 point transfer", "hyatt", "united", "ultimate rewards"]
    },
    {
        "id": "csr_high_spend_tier",
        "card": "Chase Sapphire Reserve",
        "card_code": "csr",
        "topic": "High-Spend Benefits ($75,000 Threshold)",
        "category": "High-Spend Tier",
        "annual_credit_value": 750,
        "content": (
            "Spending $75,000+ in a calendar year unlocks: "
            "1) World of Hyatt Explorist status "
            "2) IHG One Rewards Diamond Elite status "
            "3) $250 credit for The Shops at Chase "
            "4) $500 Southwest Airlines Chase Travel credit "
            "5) Southwest Airlines A-List status"
        ),
        "keywords": ["$75,000 spend", "hyatt explorist", "ihg diamond", "southwest a-list", "$500 southwest credit"]
    }
]


def get_all_benefits():
    """Returns all card benefits as a list of dicts."""
    return CARD_BENEFITS


def get_benefits_by_card(card_code: str):
    """Filters benefits by card code ('amex_plat' or 'csr')."""
    return [b for b in CARD_BENEFITS if b.get("card_code") == card_code]


def get_card_summary():
    """Returns basic card metadata summary."""
    return {
        "amex_plat": {
            "name": "American Express Platinum Card",
            "annual_fee": 895,
            "top_perks": [
                "$600 Hotel Credit",
                "$200 Airline Fee Credit",
                "$200 Uber Cash + $120 Uber One",
                "$400 Resy Credit",
                "$300 Digital Entertainment Credit",
                "$300 lululemon Credit",
                "Global Lounge Collection (Centurion, Priority Pass, Delta Sky Club)",
                "5X on Flights & Amex Travel Hotels"
            ]
        },
        "csr": {
            "name": "Chase Sapphire Reserve",
            "annual_fee": 795,
            "top_perks": [
                "$300 Annual Travel Credit",
                "$250 Chase Travel Hotel Credit (2026)",
                "$500 The Edit Hotel Credit",
                "$300 OpenTable Dining Credit",
                "DoorDash DashPass + $25/mo promos",
                "Apple TV+ & Apple Music ($288 value)",
                "Sapphire Lounge & Priority Pass Select (with 2 free guests)",
                "8X Chase Travel, 4X Direct Flights/Hotels, 3X Dining",
                "Points Boost up to 2.0¢/pt & 1:1 Transfer Partners"
            ]
        }
    }
