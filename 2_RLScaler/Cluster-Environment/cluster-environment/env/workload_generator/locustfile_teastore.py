import logging
from random import randint, choice, expovariate, gauss
from locust import HttpUser, task, between, constant, events
import locust
from prometheus_client import start_http_server, Counter, Gauge

# logging
logging.getLogger().setLevel(logging.INFO)

# Prometheus metrics
REQUESTS = Counter('locust_requests_total', 'Total number of requests')
FAILURES = Counter('locust_failures_total', 'Total number of failed requests')
RESPONSE_TIME = Gauge('locust_response_time_seconds', 'Response time in seconds')

class BaseUser(HttpUser):

    @task
    def load(self) -> None:
        """
        Simulates user behaviour.
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

    def visit_home(self) -> None:
        res = self.client.get('/')
        if res.ok:
            logging.info("Loaded landing page.")
            REQUESTS.inc()
        else:
            logging.error(f"Could not load landing page: {res.status_code}")
            FAILURES.inc()

    def login(self) -> None:
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

    def browse(self) -> None:
        for i in range(1, randint(2, 5)):
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

    def buy(self) -> None:
        user_data = {
            "firstname": "User",
            "lastname": "User",
            "adress1": "Road",
            "adress2": "City",
            "cardtype": "volvo",
            "cardnumber": "314159265359",
            "expirydate": "12/2050",
            "confirm": "Confirm"
        }
        buy_request = self.client.post("/cartAction", params=user_data)
        if buy_request.ok:
            logging.info(f"Bought products.")
            REQUESTS.inc()
        else:
            logging.error("Could not buy products.")
            FAILURES.inc()

    def visit_profile(self) -> None:
        profile_request = self.client.get("/profile")
        if profile_request.ok:
            logging.info("Visited profile page.")
            REQUESTS.inc()
        else:
            logging.error("Could not visit profile page.")
            FAILURES.inc()

    def logout(self) -> None:
        logout_request = self.client.post("/loginAction", params={"logout": ""})
        if logout_request.ok:
            logging.info("Successful logout.")
            REQUESTS.inc()
        else:
            logging.error(f"Could not log out - status: {logout_request.status_code}")
            FAILURES.inc()

class LowLoadUser(BaseUser):
    wait_time = constant(5)

class MediumLoadUser(BaseUser):
    wait_time = between(1, 3)

class HighLoadUser(BaseUser):
    wait_time = constant(1)

class ExponentialLoadUser(BaseUser):
    def exponential_wait_time(self):
        return expovariate(1/2)  # mean wait time of 2 seconds

    wait_time = exponential_wait_time

class GaussianLoadUser(BaseUser):
    def gaussian_wait_time(self):
        return max(0, gauss(2, 0.5))  # mean wait time of 2 seconds, stddev of 0.5 seconds

    wait_time = gaussian_wait_time

class RandomBurstsUser(BaseUser):
    wait_time = between(1, 5)
    
    @task
    def load(self):
        super().load()
        if choice([True, False]):
            for _ in range(randint(5, 20)):
                
                """
                Simulates user behaviour.
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

# Start Prometheus server to expose metrics
start_http_server(8000)

microservice_host = "teastore"
