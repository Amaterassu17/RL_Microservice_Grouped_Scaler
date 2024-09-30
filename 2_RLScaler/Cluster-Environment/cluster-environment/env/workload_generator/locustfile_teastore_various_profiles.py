import logging
from random import randint, choice, expovariate, gauss
from locust import HttpUser, task, between, constant, events
import locust
import math
from prometheus_client import start_http_server, Counter, Gauge

# logging
logging.getLogger().setLevel(logging.INFO)

# Prometheus metrics
REQUESTS = Counter('locust_requests_total', 'Total number of requests')
FAILURES = Counter('locust_failures_total', 'Total number of failed requests')
RESPONSE_TIME = Gauge('locust_response_time_seconds', 'Response time in seconds')

class UserBehavior(HttpUser):
    wait_time = between(1, 2)

    @task
    def load(self):
        """
        Simulates user behavior.
        :return: None
        """
        logging.info("Starting user.")
        self.visit_home()
        self.login()
        self.browse()
        # 50/50 chance to buy
        choice_buy = choice([True, False])
        if choice_buy:
            self.buy()
        self.visit_profile()
        self.logout()
        logging.info("Completed user.")

    def visit_home(self):
        """
        Visits the landing page.
        :return: None
        """
        res = self.client.get('/')
        if res.ok:
            logging.info("Loaded landing page.")
            REQUESTS.inc()

        else:
            logging.error(f"Could not load landing page: {res.status_code}")
            FAILURES.inc()


    def login(self):
        """
        User login with random userid between 1 and 90.
        :return: categories
        """
        res = self.client.get('/login')
        if res.ok:
            logging.info("Loaded login page.")
            REQUESTS.inc()
        else:
            logging.error(f"Could not load login page: {res.status_code}")
            FAILURES.inc()


        user = randint(1, 99)
        login_request = self.client.post("/loginAction", params={"username": user, "password": "password"})
        if login_request.ok:
            logging.info(f"Login with username: {user}")
            REQUESTS.inc()

        else:
            logging.error(f"Could not login with username: {user} - status: {login_request.status_code}")
            FAILURES.inc()

    def browse(self):
        """
        Simulates random browsing behavior.
        :return: None
        """
        for _ in range(1, randint(2, 5)):
            category_id = randint(2, 6)
            page = randint(1, 5)
            category_request = self.client.get("/category", params={"page": page, "category": category_id})
            if category_request.ok:
                logging.info(f"Visited category {category_id} on page 1")
                REQUESTS.inc()
                product_id = randint(7, 506)
                product_request = self.client.get("/product", params={"id": product_id})
                if product_request.ok:
                    logging.info(f"Visited product with id {product_id}.")
                    REQUESTS.inc()
                    cart_request = self.client.post("/cartAction", params={"addToCart": "", "productid": product_id})
                    if cart_request.ok:
                        logging.info(f"Added product {product_id} to cart.")
                        REQUESTS.inc()
                    else:
                        logging.error(f"Could not put product {product_id} in cart - status {cart_request.status_code}")
                        FAILURES.inc()
                else:
                    logging.error(f"Could not visit product {product_id} - status {product_request.status_code}")
                    FAILURES.inc()
            else:
                logging.error(f"Could not visit category {category_id} on page 1 - status {category_request.status_code}")
                FAILURES.inc()

    def buy(self):
        """
        Simulates buying products in the cart with sample user data.
        :return: None
        """
        user_data = {
            "firstname": "User",
            "lastname": "User",
            "address1": "Road",
            "address2": "City",
            "cardtype": "volvo",
            "cardnumber": "314159265359",
            "expirydate": "12/2050",
            "confirm": "Confirm"
        }
        buy_request = self.client.post("/cartAction", params=user_data)
        if buy_request.ok:
            logging.info("Bought products.")
            REQUESTS.inc()
        else:
            logging.error("Could not buy products.")
            FAILURES.inc()

    def visit_profile(self):
        """
        Visits user profile.
        :return: None
        """
        profile_request = self.client.get("/profile")
        if profile_request.ok:
            logging.info("Visited profile page.")
            REQUESTS.inc()
        else:
            logging.error("Could not visit profile page.")
            FAILURES.inc()

    def logout(self):
        """
        User logout.
        :return: None
        """
        logout_request = self.client.post("/loginAction", params={"logout": ""})
        if logout_request.ok:
            logging.info("Successful logout.")
            REQUESTS.inc()
        else:
            logging.error(f"Could not log out - status: {logout_request.status_code}")
            FAILURES.inc()


start_http_server(8000)



# import logging
# from random import randint, choice
# from locust import HttpUser, task, between, events
# import requests

# # Logging
# logging.getLogger().setLevel(logging.INFO)

# # Central Prometheus server endpoint
# PROMETHEUS_SERVER_URL = "http://localhost:8000"

# class UserBehavior(HttpUser):
#     wait_time = between(1, 2)

#     @task
#     def load(self):
#         self.record_request("load")
#         logging.info("Starting user.")
#         self.visit_home()
#         self.login()
#         self.browse()
#         if choice([True, False]):
#             self.buy()
#         self.visit_profile()
#         self.logout()
#         logging.info("Completed user.")

#     def visit_home(self):
#         self.record_request("visit_home")
#         res = self.client.get('/')
#         self.record_response("visit_home", res)

#     def login(self):
#         self.record_request("login")
#         res = self.client.get('/login')
#         self.record_response("login", res)
#         user = randint(1, 99)
#         login_request = self.client.post("/loginAction", params={"username": user, "password": "password"})
#         self.record_response("loginAction", login_request)

#     def browse(self):
#         self.record_request("browse")
#         for _ in range(1, randint(2, 5)):
#             category_id = randint(2, 6)
#             page = randint(1, 5)
#             category_request = self.client.get("/category", params={"page": page, "category": category_id})
#             self.record_response("browse_category", category_request)
#             if category_request.ok:
#                 product_id = randint(7, 506)
#                 product_request = self.client.get("/product", params={"id": product_id})
#                 self.record_response("browse_product", product_request)
#                 if product_request.ok:
#                     cart_request = self.client.post("/cartAction", params={"addToCart": "", "productid": product_id})
#                     self.record_response("add_to_cart", cart_request)

#     def buy(self):
#         self.record_request("buy")
#         user_data = {
#             "firstname": "User",
#             "lastname": "User",
#             "address1": "Road",
#             "address2": "City",
#             "cardtype": "volvo",
#             "cardnumber": "314159265359",
#             "expirydate": "12/2050",
#             "confirm": "Confirm"
#         }
#         buy_request = self.client.post("/cartAction", params=user_data)
#         self.record_response("buy", buy_request)

#     def visit_profile(self):
#         self.record_request("visit_profile")
#         profile_request = self.client.get("/profile")
#         self.record_response("visit_profile", profile_request)

#     def logout(self):
#         self.record_request("logout")
#         logout_request = self.client.post("/loginAction", params={"logout": ""})
#         self.record_response("logout", logout_request)

#     def record_request(self, request_type):
#         requests.post(f"{PROMETHEUS_SERVER_URL}/metrics/request", json={"type": request_type})

#     def record_response(self, request_type, response):
#         if response.ok:
#             requests.post(f"{PROMETHEUS_SERVER_URL}/metrics/success", json={"type": request_type, "latency": response.elapsed.total_seconds()})
#         else:
#             requests.post(f"{PROMETHEUS_SERVER_URL}/metrics/failure", json={"type": request_type})

# @events.request.add_listener
# def on_request(request_type, name, response_time, response_length, response):
#     requests.post(f"{PROMETHEUS_SERVER_URL}/metrics/request", json={
#         "type": request_type,
#         "name": name,
#         "response_time": response_time,
#         "response_length": response_length,
#         "response_status": response.status_code
#     })
#     if response.ok:
#         requests.post(f"{PROMETHEUS_SERVER_URL}/metrics/success", json={
#             "type": request_type,
#             "name": name,
#             "response_time": response_time,
#             "response_length": response_length,
#             "response_status": response.status_code
#         })
#     else:
#         requests.post(f"{PROMETHEUS_SERVER_URL}/metrics/failure", json={
#             "type": request_type,
#             "name": name,
#             "response_time": response_time,
#             "response_length": response_length,
#             "response_status": response.status_code
#         })
    