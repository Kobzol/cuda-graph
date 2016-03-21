#pragma once

#include <iostream>
#include <string>

#include <Windows.h>

class Timer
{
public:
	Timer()
	{
		this->start();
	}

	void start()
	{
		this->timer = GetTickCount();
	}
	size_t get_ticks()
	{
		return this->timer;
	}
	void print(std::string note="")
	{
		std::cout << note << ": " << GetTickCount() - this->timer << std::endl;
	}

private:
	size_t timer = 0;
};
