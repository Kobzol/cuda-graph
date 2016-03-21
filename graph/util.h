#pragma once

#include <iostream>
#include <string>
#include <chrono>

class Timer
{
public:
	Timer()
	{
		this->start();
	}

	long get_millis()
	{
		auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - this->timer);
		return (long) timeDiff.count();
	}

	void start()
	{
		this->timer = std::chrono::steady_clock::now();
	}
	void print(std::string note="")
	{
		std::cout << note << ": " << this->get_millis() << std::endl;
	}

private:
	std::chrono::time_point<std::chrono::system_clock> timer;
};
