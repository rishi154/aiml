from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import random

class CreditCardNumberGenerator(BaseTool):
    name: str = "Card Number Generator"
    description: str = (
        "Used for Credit Card Number Generation."
    )

    def _run(self, provider: str) -> str:
        if provider == "visa":
            # Visa cards start with 4 and have 16 digits
            prefix = random.choice([4])
            return self.generate_card_number(prefix, 16)

        elif provider == "mastercard":
            # MasterCard cards start with 51-55 and have 16 digits
            prefix = random.choice([51, 52, 53, 54, 55])
            return self.generate_card_number(prefix, 16)

        elif provider == "amex":
            # American Express cards start with 34 or 37 and have 15 digits
            prefix = random.choice([34, 37])
            return self.generate_card_number(prefix, 15)

        else:
            raise ValueError("Unsupported provider. Supported providers: visa, mastercard, amex")

        # Generate valid card number using the Luhn algorithm
    def generate_card_number(self,prefix, length):
        # Generate the first part of the card number based on prefix
        card_number = [int(digit) for digit in str(prefix)]

        # Fill the card number with random digits until the second to last digit
        while len(card_number) < length - 1:
            card_number.append(random.randint(0, 9))

        # Compute the last digit using the Luhn algorithm to make the number valid
        check_sum = 0
        for i in range(len(card_number)):
            digit = card_number[i]
            if (len(card_number) - i) % 2 == 0:
                digit *= 2
                if digit > 9:
                    digit -= 9
            check_sum += digit

        last_digit = (10 - check_sum % 10) % 10
        card_number.append(last_digit)

        return ''.join(str(digit) for digit in card_number)

    # Luhn algorithm to validate and compute the last digit (checksum)

    def luhn_check(self,card_number):
        digits = [int(d) for d in str(card_number)]
        checksum = 0
        reverse_digits = digits[::-1]
        for i, digit in enumerate(reverse_digits):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0