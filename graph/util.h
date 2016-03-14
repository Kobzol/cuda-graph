#pragma once

#include <iostream>
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
	void print()
	{
		std::cout << GetTickCount() - this->timer << std::endl;
	}

private:
	size_t timer = 0;
};
