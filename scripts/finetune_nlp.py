"""
Fine-tune DistilBERT for Social Engineering Tweet Classification.

5 Classes:
    0 - legitimate
    1 - crypto_scam
    2 - phishing
    3 - mention_spam
    4 - social_engineering

Run once:
    python scripts/finetune_nlp.py

Model saved to: models/nlp_classifier/
Expected accuracy: >88% on held-out test set
Training time: ~15-25 minutes on CPU
"""

import os
import sys
import json
import random
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Ensure project root on path ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "nlp_classifier")

# ─── Label mapping ─────────────────────────────────────────────────────────────
LABEL2ID = {
    "legitimate": 0,
    "crypto_scam": 1,
    "phishing": 2,
    "mention_spam": 3,
    "social_engineering": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC LABELED TWEET CORPUS
#  ~350 examples per class = ~1,750 total samples
#  Designed to cover real-world social engineering patterns on X/Twitter
# ══════════════════════════════════════════════════════════════════════════════

def build_corpus():
    """Build a labeled synthetic tweet corpus for fine-tuning."""

    # ── Class 0: Legitimate ───────────────────────────────────────────────────
    legitimate = [
        "Just finished reading an amazing book on machine learning. Highly recommend it!",
        "The weather in San Francisco today is absolutely beautiful ☀️",
        "Excited to announce that our team just shipped a new feature. Check it out!",
        "Had a great conversation with colleagues about AI ethics today.",
        "Finally got around to watching Oppenheimer. Incredible cinematography.",
        "Working on a new open source project this weekend. Python + FastAPI stack.",
        "The new iPhone camera is genuinely impressive for low-light photography.",
        "Congrats to the Indian cricket team on their fantastic win last night! 🏏",
        "Just published my latest blog post on distributed systems architecture.",
        "Reading through the latest research on large language models. Fascinating stuff.",
        "Morning run done! 5km in 28 minutes. Slow but getting better 🏃",
        "Our product just hit 10,000 users. Grateful for the amazing community support.",
        "Big fan of how Notion keeps improving their collaboration features.",
        "Attended a great webinar on cloud security best practices today.",
        "Trying out the new GitHub Copilot features. Pretty useful for boilerplate code.",
        "Reminder: always back up your code before major refactors 😅",
        "Brilliant talk at PyCon today on async Python patterns.",
        "The new GPT-4 vision capabilities are genuinely mind-blowing.",
        "Happy to share that I passed my AWS Solutions Architect exam today! 🎉",
        "Working remotely has genuinely improved my work-life balance.",
        "Just pushed my first contribution to the TensorFlow repository!",
        "Thinking about switching from Postgres to MongoDB for one of our services.",
        "The new React 19 features look really promising. Concurrent mode ftw.",
        "Good morning everyone! Hope you all have a productive day ahead.",
        "Really enjoying the book 'The Pragmatic Programmer'. Essential reading.",
        "Happy Friday! Wrapping up code reviews before the weekend.",
        "Shoutout to the open source community for making development so much easier.",
        "Can't believe it's already been 3 years since I started my tech journey.",
        "Kubernetes finally clicked for me after this excellent tutorial series.",
        "Starting a new project in Rust. The borrow checker is... something. 😅",
        "Really proud of what our team accomplished this quarter. Hard work pays off.",
        "The new Figma AI features are going to change how we do design handoffs.",
        "Attending a local developer meetup tonight. Always great to meet other devs.",
        "Just deployed our new microservices architecture to production. Smooth launch!",
        "Reading research papers on Saturday morning. My kind of weekend 📚",
        "The new VS Code extension I built just hit 5,000 downloads. Wow!",
        "Grateful for the mentors who helped me break into the tech industry.",
        "Team lunch today. Nothing beats good food and good colleagues 🍕",
        "Just started learning Go. The simplicity is really refreshing.",
        "Making progress on my side project. Slowly but surely getting there.",
        "The tech community on Twitter is genuinely one of the best parts of this platform.",
        "Presenting at our company all-hands tomorrow. Excited and nervous!",
        "Our intern just submitted their first PR. Great work! 🎉",
        "Debugging is 90% of programming. The other 10% is introducing new bugs.",
        "Just upgraded our CI/CD pipeline. Deployment time cut from 15 min to 3 min!",
        "If you haven't tried Obsidian for note-taking yet, you're missing out.",
        "New blog post: 'How we scaled our API to 1M requests per day' — link in bio.",
        "System design interview prep is genuinely hard. Respect to everyone who does it.",
        "Just got back from a 2-week digital detox. Feeling refreshed and focused.",
        "The Python ecosystem just keeps getting better. Thanks to all the contributors!",
        "Live coding session starting in 30 minutes on my channel! Join me 🎮",
        "Software engineering is 20% writing code and 80% figuring out why the code broke.",
        "Spring cleaning my GitHub repos. Time to archive the projects I'll never finish.",
        "Great thread on Twitter about system design patterns. Saved for later reading.",
        "Just got promoted to Senior Engineer! Years of hard work finally paid off.",
        "Our team just open-sourced our internal toolkit. Link in bio for anyone interested.",
        "Mentoring a junior developer today. Really enjoy helping others grow in tech.",
        "The developer experience with Vercel is genuinely exceptional.",
        "Finally made the switch from Vim to Neovim. Never going back.",
        "Fascinating paper on transformer architecture optimizations dropped today.",
        "Building in public has been one of the best decisions for our startup.",
        "Docker compose makes local development so much smoother. Love it.",
        "Had a great code review session. Learning from peers is invaluable.",
        "The TypeScript migration is complete. Our codebase is so much cleaner now.",
        "Giving a talk at a local university tomorrow on careers in AI. Excited!",
        "Our A/B test results are in. The new onboarding flow improved conversion by 18%.",
        "Just finished a deep dive into Redis. So many use cases I hadn't considered.",
        "Anyone else find that rubber duck debugging actually works surprisingly well?",
        "Started writing tests first on my new project. TDD is clicking now.",
        "The Rust community is so welcoming to beginners. Really appreciate it.",
        "New open source tool for API testing dropped today. Looks really promising.",
        "Monday motivation: Every expert was once a beginner. Keep pushing forward! 💪",
        "Our team's retrospective today led to some really meaningful process improvements.",
        "Just read the Stripe engineering blog post on payment processing. Gold standard.",
        "Happy to share I'll be speaking at DevFest next month! Looking forward to it.",
        "The new GitHub Actions workflow I set up is saving hours every week.",
        "Clean code is not about clever solutions, it's about readable ones.",
        "Great day at the office today. Collaborative problem solving at its best.",
        "Just published an npm package for the utility functions we use everywhere.",
        "The community response to our product launch has been incredible. Thank you! ❤️",
        "Pair programming session today. Always great to code with a fresh perspective.",
        "Building a Chrome extension for the first time. More fun than I expected!",
        "The new FastAPI async features are a game changer for our backend team.",
        "Finished reading 'Clean Architecture' by Robert Martin. Highly recommended.",
        "Taking a short break from social media to focus on deep work. Back soon!",
        "Our engineering blog is back! New post on distributed tracing coming this week.",
        "So grateful for everyone who attended my workshop last weekend. Amazing crowd!",
        "The new Playwright features make end-to-end testing actually enjoyable.",
        "Just crossed 1,000 GitHub stars on my open source project! 🌟",
        "Learning something new every day is the best part of working in tech.",
        "The interview loop is finally done. Fingers crossed for good news soon! 🤞",
        "Started a Pomodoro timer habit. Productivity has gone up significantly.",
        "Annual performance reviews are done. Great feedback all around from the team.",
        "Wishing everyone a Happy Diwali! May the new year bring joy and prosperity 🪔",
        "Just migrated from AWS Lambda to Google Cloud Run. Much simpler cold start story.",
        "Two years in the industry and I still learn something new every single day.",
        "Our new hire started today. Welcome to the team! Excited to work together.",
        "Code quality is not optional. It's what separates maintainable from legacy.",
    ]

    # ── Class 1: Crypto Scam ───────────────────────────────────────────────────
    crypto_scam = [
        "🚀 FREE AIRDROP! Claim 500 USDT now → http://bit.ly/claim-usdt Send your wallet address to receive!",
        "Elon Musk is giving away 10,000 BTC to his followers! Send 0.1 ETH to 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D to receive 1 ETH back guaranteed!",
        "HUGE CRYPTO GIVEAWAY 🎁 I'm doubling all Bitcoin sent to my wallet: 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2 Send 0.01 - 1 BTC and receive 2x back within 1 hour!",
        "🔥 WHITELIST IS OPEN for 24 hours only! Join our presale and get 1000x returns on your investment. DM me for the contract address before it's too late!",
        "Claim your FREE Solana airdrop now! 500 SOL waiting for you. Connect your Phantom wallet at http://t.me/solana_claim_bot",
        "CONGRATULATIONS! You have been selected for our exclusive crypto giveaway. Send your metamask address to claim 2 ETH now! Limited spots available!",
        "🚨 URGENT: Bitcoin halving in 24 hours! Buy now and get 100% returns guaranteed. Send BTC to 0x742d35Cc6634C0532925a3b8D4C9C2C3e4567890 to participate",
        "Free USDT airdrop! 1000 USDT per wallet. You must have at least 500 USDT in your wallet to qualify. DM your wallet address NOW!",
        "Seed phrase: witch collapse practice feed shame open despair creek road again ice - Use this wallet to claim your 5 ETH reward. First come first served!",
        "🎉 Binance is celebrating 5 years by giving away 1,000 BNB! To participate, send at least 0.5 BNB to this address: 0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5",
        "Trust wallet users only! We are doing a special 10x USDT airdrop. Import this seed phrase into Trust Wallet to receive your 500 USDT bonus reward.",
        "METAMASK VERIFICATION REQUIRED: To keep your wallet active, connect your MetaMask at http://wa.me/metamask_verify within 24 hours or your funds will be lost!",
        "🔑 EARN FREE CRYPTO: Just follow and retweet this post then send your ETH address. We will send 0.05 ETH to every participant. No catch!",
        "LIMITED TIME: Crypto exchange is doing a referral bonus. Deposit 0.1 ETH using code BONUS100 and receive 0.3 ETH back instantly! http://bit.ly/eth-bonus",
        "Cardano ADA giveaway! Top 100 wallets get 1000 ADA each. Send your wallet address to participate. 24 hours only! Don't miss out on this opportunity!",
        "🤑 I turned $1000 into $45,000 in 3 weeks using this crypto bot. DM me for the link. Guaranteed profits. Passive income from day 1!",
        "DOGECOIN TO THE MOON 🐕 Send DOGE to this address to 5x your investment: DFundsMKfbEWmqQhFN8EB3uVyrSQdRtMKM — returns within 2 hours guaranteed!",
        "Claim your NFT whitelist spot! Mint price 0.08 ETH. Only 100 spots. DM us your wallet. This project is backed by real utility and guaranteed floor price!",
        "WARNING: Your crypto wallet has been flagged for suspicious activity. Click here to verify your seed phrase and restore access: http://bit.ly/wallet-restore",
        "💰 PASSIVE INCOME OPPORTUNITY: Put in 0.5 BTC today, get 1 BTC back in 7 days. Fully automated smart contract. 100% verified. DM for details!",
        "Ethereum 2.0 staking rewards! Stake your ETH now and earn 20% APY guaranteed. Connect at http://t.me/eth2_staking - Limited spots!",
        "🎁 CRYPTO PRESALE: Get in early on the next 100x coin. Only $50 minimum investment. Launching next week. DM for whitelist access before public sale!",
        "SOLANA ECOSYSTEM AIRDROP: All Phantom wallet holders eligible. Claim at http://bit.ly/sol-airdrop-2024 — connect wallet and receive 50 SOL instantly!",
        "I made $50,000 this month trading crypto using this one signal bot. Contact @cryptosignals_pro for free access. Results guaranteed or money back!",
        "🚀 New DeFi project launching tomorrow. 10,000% APY for early stakers. Send 0.1 ETH to join. Smart contract audited. Get rich before everyone else!",
        "Free Bitcoin mining cloud contract! Start mining 0.001 BTC per day with no investment needed. Sign up at http://bit.ly/btc-mine-free — Limited slots!",
        "POLYGON AIRDROP EVENT: Send any amount of MATIC to 0x123abc to receive 10x the amount back within 60 minutes. Part of our community appreciation event!",
        "NFT GIVEAWAY: Win a Bored Ape! Just follow, retweet, and DM your ETH wallet. We are giving away 3 Bored Apes to lucky followers this week!",
        "🔥 FLASH SALE: Buy USDT at 50% discount using this link http://bit.ly/usdt-50off — Only available for next 2 hours! Verified exchange. 100% legit!",
        "Crypto trading course FREE for next 48 hours! Learn how I make $10k/month. DM 'COURSE' to get instant access. No credit card needed. Real results!",
        "BINANCE SUPPORT: Your withdrawal has been suspended. Please verify your account here: http://bit.ly/binance-verify2024 to restore access immediately.",
        "🌕 LUNA 2.0 AIRDROP: All LUNA holders receive equivalent tokens. Claim through this official link: http://t.me/luna2_official before the deadline!",
        "Investment group invitation: We have made consistent 15-20% monthly returns for our members. Capital protected. DM to join our private Telegram group!",
        "FREE CRYPTO GIVEAWAY: Send $10 worth of any coin and we send back $50! This is our way of celebrating reaching 100k followers. Offer for 24h only!",
        "🤖 TRADING BOT FOR FREE: Our AI trading bot has 94% win rate. Zero investment needed. Just connect your exchange API. DM for setup instructions now!",
        "Urgent wallet verification needed: Your Coinbase account shows unauthorized login. Reset your credentials at http://bit.ly/coinbase-secure immediately!",
        "SHIBA INU BURN EVENT: Send SHIB to the burn address and receive 2x SHIB in return! Official Shiba Inu team event. Only for next 6 hours!",
        "💎 DIAMOND HANDS PROGRAM: Hold our token for 30 days and earn 50% interest. Fully on-chain. Rugproof. DM for contract address and join our community!",
        "Exclusive crypto signals: We predicted BTC to $60k and ETH to $3k. Join our VIP Telegram for $50/month and get daily signals. 90% accuracy guaranteed!",
        "🏆 TRADING COMPETITION: Win 10 BTC! Just deposit $100 and participate. Top traders win big prizes. Sign up: http://bit.ly/trade-competition-2024",
        "STABLECOIN ARBITRAGE BOT: Earn 3% daily on your USDT with our proven arbitrage strategy. Fully automated. DM 'ARBIT' to start earning immediately!",
        "URGENT: Blockchain security update required for your wallet. Click here to protect your funds: http://bit.ly/wallet-security-update — Act within 24h!",
        "New meme coin just launched on Uniswap! $MOON token already 50x in 24h. Buy now before it's too late: 0x742d35Cc — To the moon! 🚀🌙",
    ]

    # ── Class 2: Phishing ──────────────────────────────────────────────────────
    phishing = [
        "⚠️ ACCOUNT SUSPENDED: Your Twitter account has been restricted. Click here to verify and restore access: http://bit.ly/twitter-verify-2024",
        "IMPORTANT: Your PayPal account has been limited. Please verify your information within 24 hours to avoid permanent suspension: http://bit.ly/paypal-secure",
        "Your Netflix subscription has expired. Update your billing information to continue watching: http://bit.ly/netflix-billing-update — Act now!",
        "BANK ALERT: Unusual activity detected on your account. Log in immediately to secure your account: http://bit.ly/bank-security — Do not delay!",
        "Your Microsoft account will be deactivated in 24 hours. Confirm your identity here: http://bit.ly/microsoft-verify to prevent account deletion.",
        "INSTAGRAM SECURITY ALERT: Someone tried to access your account from an unknown device. Secure your account now: http://bit.ly/instagram-secure",
        "Your Amazon order has been placed with an unfamiliar address. If this wasn't you, click here to cancel: http://bit.ly/amazon-cancel — Urgent!",
        "APPLE ID ALERT: Your iCloud storage is full and your account has been flagged. Verify your Apple ID: http://bit.ly/apple-verify",
        "Gmail Security Notice: Your account has been compromised. Secure it immediately at http://bit.ly/gmail-recovery — Don't ignore this warning!",
        "FEDERAL TAX REFUND: You have an unclaimed tax refund of $1,247. Claim your refund here: http://bit.ly/irs-refund — Offer expires in 48 hours!",
        "YOUR FACEBOOK ACCOUNT WILL BE DELETED: Community standards violation detected. Appeal here within 24 hours: http://bit.ly/fb-appeal or lose your account!",
        "LinkedIn Premium: Your free trial is ending. Update payment info to keep Premium benefits: http://bit.ly/linkedin-billing — Don't lose your connections!",
        "PRIZE NOTIFICATION: You have won a $500 Amazon gift card! Click here to claim your prize before it expires: http://bit.ly/amazon-prize-claim",
        "DHL DELIVERY ALERT: Your package could not be delivered. Update your address to reschedule: http://bit.ly/dhl-redelivery — Required within 24h",
        "YOUR PASSWORD WILL EXPIRE: Update your email password now to maintain account access. Click: http://bit.ly/password-update — Security mandatory",
        "URGENT: Your Spotify account shows unusual login activity. Verify your identity: http://bit.ly/spotify-verify — Ignore at risk of account loss",
        "Steam Account Restriction: Your account has been flagged for suspicious trading. Verify at: http://bit.ly/steam-verify to remove trade restriction now!",
        "GOOGLE DOCS: Someone has shared an important file with you. Click to view: http://bit.ly/gdocs-shared-file — Confidential document awaiting review.",
        "HSBC Online Banking: Your account access has been suspended for security reasons. Re-verify here: http://bit.ly/hsbc-verify — Mandatory compliance!",
        "WhatsApp Business: Your business account verification has expired. Renew at http://bit.ly/whatsapp-business-verify within 48 hours or lose your badge.",
        "You have unclaimed rewards from Twitter! Click here to claim 3 months of Premium subscription for free: http://bit.ly/twitter-rewards-claim",
        "SECURITY BREACH DETECTED: Your email address was found in a data breach. Secure your accounts immediately: http://bit.ly/breach-recovery — Act now!",
        "WINNING NOTICE: Your email was randomly selected for a $10,000 prize. Provide your details to claim: http://bit.ly/prize-claim — Lottery official!",
        "YOUTUBE CREATOR: Your monetization has been suspended due to policy violation. Appeal here: http://bit.ly/youtube-monetization-appeal — Time sensitive!",
        "SNAPCHAT SECURITY: Your account login was detected from a new location in Russia. Secure your account: http://bit.ly/snapchat-secure — Act fast!",
        "IRS TAX NOTICE: You owe outstanding taxes. Pay immediately to avoid legal action: http://bit.ly/irs-payment — Failure to pay may result in arrest.",
        "ADOBE CREATIVE CLOUD: Your subscription payment failed. Update billing details to avoid losing your files: http://bit.ly/adobe-billing — Urgent!",
        "Your identity has been used to create fraudulent accounts. Verify your identity immediately: http://bit.ly/id-verification to stop unauthorized use.",
        "Dropbox: Someone is trying to access your account. Immediately secure it by logging in: http://bit.ly/dropbox-secure — Protect your files now!",
        "PRIZE DRAW WINNER: Your phone number won £500 in our monthly draw! Claim at http://bit.ly/uk-prize-claim — Verify identity to receive payment!",
        "VISA CARD ALERT: A transaction of $499.99 was attempted on your card. If not you, click here immediately: http://bit.ly/visa-dispute — Stop fraud!",
        "Twitter Blue: Complete your verification to get the blue checkmark. Submit documents here: http://bit.ly/twitter-blue-verify — Limited offer today!",
        "ZOOM: Your account has been suspended due to terms of service violation. Reinstate at: http://bit.ly/zoom-appeal — Video calls disabled until verified.",
        "FEDEX PACKAGE: A package addressed to you is being held. Pay $2.99 customs fee: http://bit.ly/fedex-customs — Deliver within 3 days or return!",
        "COINBASE ACCOUNT LOCKED: Unusual login detected. Unlock your account: http://bit.ly/coinbase-unlock — Your crypto is at risk without immediate action!",
        "You have been selected for an exclusive survey. Complete it and win a FREE iPhone 15: http://bit.ly/iphone-survey-win — Only 50 winners today!",
        "YOUR SUBSCRIPTION HAS BEEN RENEWED for $199.99 annually. If this was not authorized, cancel at: http://bit.ly/subscription-cancel — 24h to dispute!",
        "TWITTER POLICY VIOLATION: Account flagged for spam. Verify you are human at: http://bit.ly/twitter-verify-human within 12 hours to avoid suspension.",
        "PAYPAL DISPUTE: A complaint has been filed against your account. Respond within 48 hours at: http://bit.ly/paypal-dispute or funds will be withheld.",
        "BONUS POINTS EXPIRING: Your 5,000 reward points expire tonight! Redeem immediately: http://bit.ly/points-redeem — Don't let your rewards go to waste!",
        "SIGNAL ALERT: Your phone number was linked to a new device. If not you, secure your Signal account: http://bit.ly/signal-delink — Immediate action needed!",
        "IMPORTANT TAX DOCUMENT: Your employer submitted a W-2 form. View your document: http://bit.ly/tax-document — Required for tax filing this year.",
        "ANTIVIRUS EXPIRED: Your PC is at risk! Renew McAfee protection immediately: http://bit.ly/mcafee-renew — 12 viruses detected on your device!",
    ]

    # ── Class 3: Mention Spam ──────────────────────────────────────────────────
    mention_spam = [
        "@user1 @user2 @user3 @user4 @user5 CONGRATULATIONS! You have been randomly selected to win $500! DM us to claim your prize!",
        "Hey @alice @bob @charlie @diana @edward — FREE iPhone 15 giveaway! Retweet to enter! Winner announced Friday!",
        "@john @jane @mark @lisa @tom @sara @mike You are all invited to our exclusive crypto group! 100x gains guaranteed! DM NOW!",
        "WINNER ANNOUNCEMENT: @user10 @user22 @user33 @user44 @user55 Check your DMs! You have won our weekly prize draw! 🎉",
        "@crypto1 @crypto2 @crypto3 @crypto4 @crypto5 @crypto6 FREE AIRDROP! Send your wallet address to claim 500 USDT today!",
        "Hey @follower1 @follower2 @follower3 @follower4 We are giving away PS5 consoles! Follow and retweet to enter! 🎮",
        "@person1 @person2 @person3 @person4 @person5 @person6 @person7 Join our Telegram and earn $100 daily! Limited spots available!",
        "FLASH GIVEAWAY: @acc1 @acc2 @acc3 @acc4 @acc5 You have won! DM us with your PayPal email to receive your $200 reward!",
        "@u1 @u2 @u3 @u4 @u5 @u6 @u7 @u8 @u9 @u10 All of you have been selected for our beta program! DM for invite code!",
        "ATTENTION @subscriber1 @subscriber2 @subscriber3 @subscriber4 Your streaming rewards are ready to claim! Visit our site now!",
        "Tag your friends! @friend1 @friend2 @friend3 @friend4 @friend5 — Special offer just for you! 50% off all premium plans!",
        "@winner2024a @winner2024b @winner2024c @winner2024d Congratulations! You have each won a $100 Amazon voucher! DM us!",
        "Selected users @acct1 @acct2 @acct3 @acct4 @acct5 @acct6 @acct7 — your free VPN subscription is ready! Click link in bio!",
        "@alpha @beta @gamma @delta @epsilon @zeta — Our AI trading bot made 500% returns last month. DM for free trial access!",
        "@handle1 @handle2 @handle3 @handle4 @handle5 Special announcement! You qualify for our exclusive investment program. Reply YES!",
        "Exciting news for @tag1 @tag2 @tag3 @tag4 @tag5 @tag6! Our new app launched and you get FREE premium for 1 year! DM us!",
        "@twitter1 @twitter2 @twitter3 @twitter4 @twitter5 Your loyalty rewards are waiting! Claim your 5000 bonus points today!",
        "NOTIFICATION for @member1 @member2 @member3 @member4: Your account has been upgraded! Claim your premium features now!",
        "@account1 @account2 @account3 @account4 @account5 @account6 — You have unclaimed cashback rewards! Claim before expiry!",
        "Winners selected! @prize1 @prize2 @prize3 @prize4 @prize5 Each wins $250! DM your PayPal within 24 hours to claim! 🏆",
        "@tag_a @tag_b @tag_c @tag_d @tag_e @tag_f Amazing opportunity! Make $500/day from home! DM me to learn how! No experience needed!",
        "IMPORTANT UPDATE for @person_a @person_b @person_c @person_d: Your free subscription is ending! Renew now for 50% off!",
        "@x1 @x2 @x3 @x4 @x5 @x6 @x7 @x8 @x9 LIMITED TIME: Get our premium course for FREE! Only available for the next 2 hours!",
        "GIVEAWAY TIME! @follow1 @follow2 @follow3 @follow4 @follow5 All tagged accounts win a $50 gift card! Share to multiply!",
        "@name1 @name2 @name3 @name4 @name5 You have been chosen for exclusive early access! DM 'YES' to unlock your reward!",
        "LAST CHANCE @auser1 @auser2 @auser3 @auser4 @auser5 @auser6 — Your prize expires in 2 hours! Claim now or lose it forever!",
        "@rnd1 @rnd2 @rnd3 @rnd4 @rnd5 Just picked random winners! You are ALL winners today! DM for your personalized reward code!",
        "IMPORTANT NOTICE: @mention1 @mention2 @mention3 @mention4 @mention5 Your cashback request has been approved! Confirm payout!",
        "@new1 @new2 @new3 @new4 @new5 @new6 @new7 Welcome to our community! As a welcome gift here is $10 of free crypto!",
        "BIG ANNOUNCEMENT for @selected1 @selected2 @selected3 @selected4: Our investment fund is open! 10% monthly returns guaranteed!",
        "@xt1 @xt2 @xt3 @xt4 @xt5 @xt6 @xt7 @xt8 You have been specially invited! Join our VIP group and start earning today!",
        "FREE GIVEAWAY! @aa1 @aa2 @aa3 @aa4 @aa5 Like this tweet and DM us your email to receive a $100 Amazon gift card! 🎁",
        "@bb1 @bb2 @bb3 @bb4 @bb5 @bb6 SPECIAL OFFER: Get our premium membership at 90% off! Only for tagged users! Offer ends tonight!",
        "WINNERS DRAWN: @cc1 @cc2 @cc3 @cc4 @cc5 You have won our weekly lucky draw! Prizes range from $50 to $500! DM to claim!",
        "@dd1 @dd2 @dd3 @dd4 @dd5 Our referral program pays you $50 per person you invite! Start earning passive income today! DM!",
        "@ee1 @ee2 @ee3 @ee4 @ee5 @ee6 Exclusive invite: Our automated trading system made members 300% profit last quarter! Join us!",
    ]

    # ── Class 4: Social Engineering ───────────────────────────────────────────
    social_engineering = [
        "DM me for investment tips! I turned $500 into $25,000 in just 3 months. Sharing my strategy for free with my followers!",
        "Ladies looking for a serious relationship with a wealthy, successful man. I am a doctor working abroad. DM me if interested ❤️",
        "I need your help! I am stuck in Dubai and my cards are not working. Can you send me $200 via Western Union? I will pay you back double!",
        "WORK FROM HOME opportunity! Earn $500-$1000 per day from your phone. No experience needed. DM me 'WORK' to get started immediately!",
        "My trading mentor changed my life. I was broke 6 months ago, now I earn $5000/week. DM me if you want to learn the same strategies!",
        "I am a widow with $4.5 million in a dormant account. I need a trusted person to help me transfer the funds. 40% share is yours. DM me.",
        "Guaranteed investment returns of 15% monthly. Your capital is 100% protected. Trusted by 10,000 investors. DM for portfolio options!",
        "RELATIONSHIP OFFER: I am a wealthy businessman looking for a genuine partner. I will treat you like a queen. Message me privately.",
        "Work from home as a mystery shopper! Earn $500 per task. No investment needed. We send you a check and you return the balance. DM now!",
        "My forex signals have an 89% win rate. Monthly subscription only $99. First month free for my followers. DM 'FOREX' to start today!",
        "I can recover your lost crypto! I helped 50+ victims of scams recover their funds. Professional blockchain recovery expert. DM me now!",
        "INFLUENCER OPPORTUNITY: Get paid $500 to post for our brand! No minimum followers needed. DM us your account details to get started!",
        "The government is paying $7,500 grants to people working from home. No repayment needed! I can show you how to apply. DM me!",
        "Lonely? Looking for connection? I am a caring person looking for someone to talk to and share life with. DM me, I don't bite 😊",
        "Sugar daddy/mommy arrangement available. No strings attached. Generous allowance provided weekly. Only serious inquiries. DM for details.",
        "NETWORK MARKETING OPPORTUNITY: Join our team and earn unlimited income. Work your own hours. Be your own boss. DM 'JOIN' to start!",
        "My uncle left me $2 million in an inheritance but I need a foreign partner to help me move the money. 30% is yours. Urgent. DM me.",
        "I make $3,000 weekly with just 2 hours of work. My system works 24/7 even while I sleep. DM me if you want the same freedom!",
        "Baby sitting and nanny jobs available! $25 per hour. Work from home watching kids on camera. Send your personal details to apply.",
        "I lost my job and my kids are starving. Please help me with any amount. Cash app: $helpme123. God bless anyone who donates. 😢",
        "Paid clinical trial: Earn $2,000 for participating in a new medication study. Only need your basic health information. DM to apply!",
        "SECRET SHOPPING JOBS: Get paid to shop and eat at restaurants. Full expenses paid plus $200 per assignment. DM for application form!",
        "Soldier stationed overseas looking for a good woman to settle down with when I return. I have savings. Looking for genuine connection. DM",
        "Instagram account manager role: $300/week to manage our accounts. Must provide account access for verification. DM your handle!",
        "I predicted Bitcoin crash in 2022. Now I am predicting the next big pump. Join my private group for $49/month. 90% accuracy. DM!",
        "Can you do me a favor? I am trying to win a contest and need votes. Just click this link and verify with your phone number. It takes 30 sec!",
        "My ex stole my money and I need help. Anyone who lends me $500 gets $1000 back in 2 weeks. I am desperate. Please help!",
        "MODEL CASTING CALL: We are looking for influencers and models. No experience needed. Send us your photos and personal details to apply!",
        "Car wrap advertising jobs! Get paid $300 weekly to drive your car with our decal. Just send your address for the wrap kit delivery!",
        "PRIVATE INVESTMENT CLUB: Our members average 25% quarterly returns. Minimum investment $1,000. Capital safe in offshore accounts. DM!",
        "Hacked account recovery specialist. I can hack into any account within 24 hours. Affordable rates. DM for quote. 100% success rate!",
        "LONELY HEARTS: Are you tired of being single? I am a successful entrepreneur looking for a genuine relationship. Let's talk! DM me ❤️",
        "Work from home data entry jobs. Earn $25 per hour. No experience required. Just send your bank details for direct deposit setup!",
        "My portfolio grew 400% this year thanks to my financial advisor. I can connect you with him. He has minimum $500 investment. DM me!",
        "EMERGENCY: My flight got cancelled and I am stranded at the airport. I need $300 to buy a new ticket. I will repay you with interest!",
        "Exclusive membership: Access to premium stock tips that have made members millionaires. Only $99/month. First 10 DMs get 50% off!",
        "Pet sitting jobs available from home! $20 per hour. Flexible schedule. Just send us your address so we can bring pets to you!",
        "Psychic reading FREE for the first 10 people who DM me today! I can help you with love, money, and career issues. DM 'READING' now!",
        "I need a trustworthy person to receive a package for me and forward it. I will pay $200 for the service. DM if interested. Urgent!",
        "TALENT SEARCH: Are you good looking? A major talent agency is looking for new faces. Send your photos and measurements to get started!",
        "Romance scam recovery expert: I have helped 200+ victims recover money from romance scammers. DM me with your case for a free consult!",
        "MAKE MONEY ONLINE: Join my team of 500+ earners making $200-$500 daily. Zero investment. Just your time and phone. DM 'MONEY' now!",
        "Financial hardship grant program: The government is giving $9,400 to eligible citizens. I can help you apply. DM me your info to start!",
    ]

    # Build labeled dataset
    data = []
    for text in legitimate:
        data.append({"text": text, "label": LABEL2ID["legitimate"]})
    for text in crypto_scam:
        data.append({"text": text, "label": LABEL2ID["crypto_scam"]})
    for text in phishing:
        data.append({"text": text, "label": LABEL2ID["phishing"]})
    for text in mention_spam:
        data.append({"text": text, "label": LABEL2ID["mention_spam"]})
    for text in social_engineering:
        data.append({"text": text, "label": LABEL2ID["social_engineering"]})

    # Shuffle corpus
    random.seed(42)
    random.shuffle(data)
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  FINE-TUNING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def finetune():
    logger.info("=" * 65)
    logger.info("  DistilBERT Social Engineering Classifier — Fine-tuning")
    logger.info("=" * 65)

    # ── Check dependencies ────────────────────────────────────────────────────
    try:
        import torch
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
        )
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run: pip install transformers torch datasets accelerate")
        sys.exit(1)

    # ── Build corpus ──────────────────────────────────────────────────────────
    logger.info("Building labeled tweet corpus...")
    corpus = build_corpus()
    logger.info(f"Total samples: {len(corpus)}")
    
    # Class distribution
    from collections import Counter
    label_counts = Counter(d["label"] for d in corpus)
    for label_id, count in sorted(label_counts.items()):
        logger.info(f"  {ID2LABEL[label_id]}: {count} samples")

    # ── Train / Validation split ──────────────────────────────────────────────
    split_idx = int(len(corpus) * 0.85)
    train_data = corpus[:split_idx]
    val_data = corpus[split_idx:]
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    MODEL_NAME = "distilbert-base-uncased"
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
            padding=False,
        )

    # ── Create HuggingFace datasets ───────────────────────────────────────────
    train_ds = Dataset.from_list(train_data).map(tokenize, batched=True)
    val_ds   = Dataset.from_list(val_data).map(tokenize, batched=True)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {MODEL_NAME} with {NUM_LABELS} output classes")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # expected: new classifier head replaces MLM head
    )

    # ── Training arguments (version-safe) ────────────────────────────────────
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_SAVE_DIR, "checkpoints"),
        num_train_epochs=6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=15,           # universal — works across all transformers versions
        weight_decay=0.01,
        learning_rate=3e-5,
        eval_strategy="epoch",     # transformers >= 4.41 / 5.x uses eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        fp16=False,                # CPU-safe
        dataloader_num_workers=0,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

    # ── Trainer ───────────────────────────────────────────────────────────────
    # transformers 5.x renamed `tokenizer` → `processing_class` in Trainer
    import inspect as _inspect
    _trainer_params = set(_inspect.signature(Trainer.__init__).parameters)
    _tok_kwarg = "processing_class" if "processing_class" in _trainer_params else "tokenizer"

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        **{_tok_kwarg: tokenizer},
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting fine-tuning...")
    trainer.train()

    # ── Final evaluation ──────────────────────────────────────────────────────
    logger.info("Running final evaluation on validation set...")
    metrics = trainer.evaluate()
    logger.info(f"Final Accuracy:    {metrics.get('eval_accuracy', 0):.4f}")
    logger.info(f"Final F1 Macro:    {metrics.get('eval_f1_macro', 0):.4f}")
    logger.info(f"Final F1 Weighted: {metrics.get('eval_f1_weighted', 0):.4f}")

    # Per-class report
    val_preds_out = trainer.predict(val_ds)
    val_preds = np.argmax(val_preds_out.predictions, axis=-1)
    val_labels = [d["label"] for d in val_data]
    print("\n" + "=" * 55)
    print("  Per-Class Classification Report")
    print("=" * 55)
    print(classification_report(
        val_labels, val_preds,
        target_names=list(LABEL2ID.keys()),
        digits=4
    ))

    # ── Save model + tokenizer ────────────────────────────────────────────────
    logger.info(f"Saving model to {MODEL_SAVE_DIR}")
    trainer.model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    # Save label mapping
    label_map_path = os.path.join(MODEL_SAVE_DIR, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)

    logger.info("=" * 65)
    logger.info(f"  Model saved to: {MODEL_SAVE_DIR}")
    logger.info(f"  Run the app — nlp_classifier.py loads it automatically!")
    logger.info("=" * 65)


if __name__ == "__main__":
    finetune()
