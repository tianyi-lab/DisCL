import os
import pickle
import sys

sys.path.append('.')
import src.templates as templates

import pandas as pd
import argparse
import pdb
from typing import List, Dict
from tqdm import tqdm

openai_classnames = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
                     "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
                     "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
                     "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt",
                     "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog",
                     "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle",
                     "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama",
                     "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon",
                     "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake",
                     "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake",
                     "water snake", "vine snake", "night snake", "boa constrictor", "African rock python",
                     "Indian cobra", "green mamba", "sea snake", "Saharan horned viper",
                     "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion",
                     "yellow garden spider", "barn spider", "European garden spider", "southern black widow",
                     "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse",
                     "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw",
                     "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
                     "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna",
                     "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm",
                     "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab",
                     "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish",
                     "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo",
                     "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule",
                     "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher",
                     "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong",
                     "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu",
                     "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound",
                     "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound",
                     "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound",
                     "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki",
                     "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
                     "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier",
                     "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier",
                     "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier",
                     "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer",
                     "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier",
                     "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever",
                     "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer",
                     "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
                     "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
                     "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
                     "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
                     "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
                     "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
                     "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
                     "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
                     "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
                     "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
                     "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
                     "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
                     "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
                     "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
                     "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah",
                     "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat",
                     "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle",
                     "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect",
                     "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
                     "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly",
                     "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish",
                     "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine",
                     "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig",
                     "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)",
                     "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel",
                     "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger",
                     "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang",
                     "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus",
                     "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey",
                     "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri",
                     "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel",
                     "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish",
                     "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner",
                     "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron",
                     "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen",
                     "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn",
                     "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon",
                     "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker",
                     "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib",
                     "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh",
                     "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie",
                     "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle",
                     "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon",
                     "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
                     "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
                     "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
                     "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
                     "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
                     "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
                     "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat",
                     "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball",
                     "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper",
                     "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock",
                     "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven",
                     "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope",
                     "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck",
                     "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain",
                     "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat",
                     "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball",
                     "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille",
                     "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper",
                     "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp",
                     "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt",
                     "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron",
                     "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono",
                     "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap",
                     "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick",
                     "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass",
                     "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca",
                     "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet",
                     "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt",
                     "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery",
                     "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
                     "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
                     "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
                     "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask",
                     "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas",
                     "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter",
                     "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume",
                     "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier",
                     "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship",
                     "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger",
                     "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot",
                     "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector",
                     "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator",
                     "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel",
                     "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle",
                     "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe",
                     "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale",
                     "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt",
                     "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket",
                     "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask",
                     "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow",
                     "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl",
                     "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web",
                     "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge",
                     "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram",
                     "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses",
                     "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing",
                     "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear",
                     "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine",
                     "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole",
                     "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle",
                     "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile",
                     "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase",
                     "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin",
                     "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
                     "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
                     "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok",
                     "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
                     "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
                     "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
                     "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
                     "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
                     "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
                     "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
                     "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
                     "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser",
                     "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
                     "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
                     "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
                     "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

openai_classnames = [item.replace('/', 'or') for item in openai_classnames]
template = getattr(templates, 'openai_imagenet_template_single')


def filter_img(clip_path: str, threshold: float):
    dict_filt = dict()
    img_cnt = 0
    if os.path.exists(clip_path):
        with open(clip_path, 'rb') as f:
            dict_clip_res = pickle.load(f)
        list_filtered = list(dict_clip_res.items())
        list_filtered = [[item[0].split('='), item[1][0], item[1][1]] for item in
                         list_filtered]  # list_filtered = [[sp_name, cate, img_id], score]
        # list_filtered = [[item[0][0], item[0][1], item[0][2], item[1], item[2]] for item in list_filtered]
        print(f"length before filter: {len(list_filtered)}")
        list_filtered = [item[0] for item in list_filtered if item[1] >= threshold]
        print(f"length after filter: {len(list_filtered)}")
        for pair in list_filtered:
            cur_sp = pair[0]
            cur_cate = pair[1]
            cur_imgid = pair[2]
            if cur_cate not in dict_filt:
                dict_filt[cur_cate] = dict()
            if cur_sp not in dict_filt[cur_cate]:
                dict_filt[cur_cate][cur_sp] = []
            dict_filt[cur_cate][cur_sp].append(cur_imgid)
            img_cnt += 1
    return dict_filt, img_cnt


def merge_with_prompt(df, df_label_temp, merge_type='train'):
    # merge prompts
    if merge_type == 'train':
        # df_label_temp = df_label_temp.drop_duplicates(subset=['label'])
        df = pd.merge(df, df_label_temp, on='label', how='left')
    else:
        df_partial = df_label_temp.drop_duplicates(subset=['label'])
        df = pd.merge(df, df_partial, on='label', how='left')

    df = df[['title', 'filepath', 'label', 'strength', 'guidance', 'seed', 'img_id']]
    print(f"length of final {merge_type}.csv: {len(df)}")
    return df


def main():
    img_sp_folder = os.listdir(args.input_folder)
    img_sp_folder = [item for item in img_sp_folder]

    list_result = []
    if not args.test:
        df_train_ori = pd.read_csv(f'{args.data_folder}/data/imagenet/LT_metadata/train.csv')
        train_class_cnt = df_train_ori.groupby('label').count()['filepath'].reset_index()
        train_class_cnt = train_class_cnt.rename(columns={'filepath': 'train_cnt'})
    else:
        df_train = pd.read_csv(f'{args.data_folder}/data/imagenet/LT_metadata/train.csv')
        df_train_ori = pd.read_csv(f'{args.data_folder}/data/imagenet/LT_metadata/test.csv')
        train_class_cnt = df_train.groupby('label').count()['filepath'].reset_index()
        train_class_cnt = train_class_cnt.rename(columns={'filepath': 'train_cnt'})

    # df_train_ori.rename({'filepath': 'filename', 'label': 'y'}, axis='columns', inplace=True)
    df_train_ori['strength'] = 0
    df_train_ori['seed'] = 100
    df_train_ori = df_train_ori[['label', 'filepath', 'strength', 'seed']]
    cur_train_ori = df_train_ori.values.tolist()
    list_result.extend(cur_train_ori)
    print(f'Original data: {len(list_result)}')

    label_to_template = []
    for label, cls_name in enumerate(openai_classnames):
        for t in template:
            caption = t(cls_name)
            label_to_template.append([label, caption])
    df_label_temp = pd.DataFrame(label_to_template, columns=['label', 'title'])

    if args.curriculum:
        threshold = 0.30
        Dict_filt, img_cnt = filter_img(args.clip_score, threshold)

        all_cnt = 0
        filtered_cnt = 0
        for cur_sp_f in tqdm(img_sp_folder):
            cur_sp_path = os.path.join(args.input_folder, cur_sp_f)
            cur_sp_name = cur_sp_f.replace('_', ' ')
            list_img_cate = os.listdir(cur_sp_path)
            if len(list_img_cate) == 1:
                # have / in name, resulting in the incorrect folder structure
                post_fix = list_img_cate[0]
                cur_sp_path = os.path.join(cur_sp_path, post_fix)
                list_img_cate = os.listdir(cur_sp_path)
                cur_sp_name += '/' + post_fix.replace('_', ' ')
                cur_sp_f += '/' + post_fix

            cur_y = openai_classnames.index(cur_sp_name)

            for cate in list_img_cate:
                cur_strength = int(cate.split('_')[0].replace('Strength', ''))
                cur_seed = int(cate.split('_')[-1].replace('seed', ''))
                cur_cate_path = os.path.join(cur_sp_path, cate)
                list_sub_img = os.listdir(cur_cate_path)
                list_sub_img = [item for item in list_sub_img if 'JPEG' in item]

                for img_name in list_sub_img:
                    cur_img_path = os.path.join(cur_cate_path, img_name)
                    img_name = img_name.replace('.JPEG', '')
                    all_cnt += 1
                    if len(Dict_filt) > 0:
                        if cate in Dict_filt and cur_sp_f in Dict_filt[cate] and img_name.replace('.JPEG', '') in \
                                Dict_filt[cate][cur_sp_f]:
                            list_result.append([cur_y, cur_img_path, cur_strength, cur_seed])
                            filtered_cnt += 1

                    else:
                        list_result.append([cur_y, cur_img_path, cur_strength, cur_seed])

    #############################################
    # using all training data
    print(f'generated data: {len(list_result)}')

    df = pd.DataFrame(list_result, columns=['label', 'filepath', 'strength', 'seed'])
    df.loc[:, 'guidance'] = df['strength'].apply(lambda x: 100 - int(x))
    df.loc[:, 'img_name'] = df['filepath'].apply(lambda x: x.split('/')[-1].replace('.JPEG', ''))

    print('adding img id')
    # change img_name to int img_id
    # if img_id >= 0: enhanced data
    # if img_id < 0: data that are not enhanced
    df_count = df.groupby(['img_name']).count()['guidance']
    list_guid_img_name = list(df_count[df_count > 1].index)
    Dict_img_id = {list_guid_img_name[i]: i for i in range(len(list_guid_img_name))}

    list_ori_guid = list(df_count[df_count == 1].index)
    Dict_img_id_ori = {list_ori_guid[i]: i + 1 for i in range(len(list_ori_guid))}
    df.loc[:, 'img_id'] = df['img_name'].apply(lambda x: Dict_img_id[x] if x in Dict_img_id else -Dict_img_id_ori[x])

    if args.curriculum:
        # select image_id with all guidance
        print(f"selecting images with all guidance for guidance selection")
        df_count = df.groupby(['img_name', 'guidance']).count().reset_index()
        df_count = df_count.groupby(['img_name', ]).count()['guidance'].reset_index()
        sel_img = df_count[df_count['guidance'] == 6].sample(n=100, replace=False, random_state=42)[
            'img_name'].values.tolist()
        df_sel = df[df['img_name'].isin(sel_img)].reset_index(drop=True)
        df_sel = df_sel.groupby(['img_name', 'guidance']).apply(
            lambda x: x.sample(n=1, replace=False, random_state=42)).reset_index(drop=True)

        # # exclude validate set from training samples
        df = df[~df['img_name'].isin(sel_img)].reset_index(drop=True)

        # df_sel = df[df['img_id'] >= 0]
        # df = df[df['img_id'] < 0]

        df_sel_final = merge_with_prompt(df_sel, df_label_temp, merge_type='curriculum')
        # df_sel_final = df_sel_final[(df_sel_final['guidance'] >= 30)]
        print(f'Data for curriculum: {len(df_sel_final)}')
        df_sel_final = pd.merge(df_sel_final, train_class_cnt, on='label', how='inner')
        print(f'Data for curriculum: {len(df_sel_final)}')
        df_sel_final.to_csv(os.path.join(args.save_folder, f'curriculum.csv'), sep='\t', index=False, header=True)

    if not args.test:
        # merge prompts
        df_final = merge_with_prompt(df, df_label_temp, merge_type='train')
        # df_final = df_final[(df_final['guidance'] >= 30)]
        print(f'Data for training: {len(df_final)}')
        df_final = pd.merge(df_final, train_class_cnt, on='label', how='inner')
        print(f'Data for training: {len(df_final)}')
        df_final.to_csv(os.path.join(args.save_folder, f'train.csv'), sep='\t', index=False, header=True)
    else:
        # merge prompts
        df_final = merge_with_prompt(df, df_label_temp, merge_type='test')
        print(f'Data for test: {len(df_final)}')
        df_final = pd.merge(df_final, train_class_cnt, on='label', how='inner')
        df_final.to_csv(os.path.join(args.save_folder, f'test.csv'), sep='\t', index=False, header=True)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--curriculum', action=argparse.BooleanOptionalAction)
    parser.add_argument('--clip_score', required=True, help="Path to clip_score.pkl")
    parser.add_argument('--save_folder', required=True, help="Path to save all csv files")
    parser.add_argument('--input_folder', required=True, help='Path to synthetic data')
    parser.add_argument('--data_folder', required=True, help='Path to synthetic data')
    parser.add_argument('--test', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    # args.curriculum = True

    main()
